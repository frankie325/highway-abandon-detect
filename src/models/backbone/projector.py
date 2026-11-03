from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable, Sequence
from typing import cast

from loguru import logger


class LayerNorm(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over
    the channel dimension for inputs that have shape (batch_size, channels, height, width).

    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        # LayeNorm始终对最后len(normalized_shape)维度进行归一化
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        # 沿着最后一维通道数（C）进行层归一化
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 还原
        x = x.permute(0, 3, 1, 2)
        return x


def get_activation(name: str | None, inplace: bool = False) -> nn.Module:
    """Get activation."""
    module: nn.Module
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError(f"Unsupported act type: {name}")
    return module


class ConvX(nn.Module):
    """Conv-bn module. 卷积+归一化模块"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel: int | tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: str = "relu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)

        # (in_feature - kernel_size + 2*padding) / stride + 1
        # padding 取值为kernel_size / 2, in_feature - kernel_size + 2*padding = in_feature,
        # 说明仅仅通过stride控制特征图的缩放
        padding: tuple[int, int] = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )

        self.bn: nn.Module
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = (
                LayerNorm(out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
            )
        self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv必须接受连续的张量，因为上游（如自定义 LayerNorm 的 permute）会让特征图变成非连续视图
        return self.act(self.bn(self.conv(x.contiguous())))


class Bottleneck(nn.Module):
    """标准 bottleneck 结构"""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层层数
        self.cv1 = ConvX(
            c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.cv2 = ConvX(
            c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.add = shortcut and c1 == c2  # 输入通道和输出相等时才能做残差连接

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Yolov8 C2f模块"""

    def __init__(
        self,
        c1: int,  # 输入通道数
        c2: int,  # 输出通道数
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,  # 控制隐藏层通道数
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvX(
            c1, 2 * self.c, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.cv2 = ConvX(
            (2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.bottlenecks = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                act=act,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道数分成2份
        y = list(self.cv1(x).split((self.c, self.c), 1))  # 2个特征图
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottlenecks)  # 2 + n个特征图

        y = torch.cat(y, 1)  # 沿通道数连接: (B, 2 + n * c, H, W)
        return self.cv2(y)  # (B, c2 + n * c, H, W)


class MultiScaleProjector(nn.Module):
    def __init__(
        self,
        projector_scale: list[Literal["P3", "P4", "P5", "P6"]],
        in_channels: Sequence[
            int
        ],  # DINOv3 vitb16输出特征层的通道数: [768, 768, 768, 768]
        out_channels: int,  # 投影层输出通道数
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()
        self.use_extra_pool = False
        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "projector scale必须按照P3/P4/P5/P6升序排列"
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [(lvl, level2scalefactor[lvl]) for lvl in self.projector_scale]

        stages_sampling: list[nn.ModuleList] = []
        stages: list[nn.Sequential] = []
        stages_levels: list[str] = []
        for lvl, scale in scale_factors:
            scale_stage_layers: list[nn.Module] = []
            for in_dim in in_channels:
                layers: list[nn.Module] = []
                # 通道数缩小为1/2，特征图上采样放大2倍
                if scale == 2.0:
                    # H_out = (H_in - 1) × stride - 2 × padding + kernel_size + output_padding
                    # (H_in - 1) * 2 - 2 * 0 + 2 = 2*H_in
                    layers.extend(
                        [
                            nn.ConvTranspose2d(
                                in_dim, in_dim // 2, kernel_size=2, stride=2
                            ),
                        ]
                    )

                # 保持原样不动
                elif scale == 1.0:
                    pass
                # 通道数不变, 特征图下采样缩小为1/2
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(
                                in_dim,
                                in_dim,
                                3,
                                2,
                                layer_norm=layer_norm,
                                rms_norm=rms_norm,
                            )
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError(f"Unsupported scale_factor:{scale}")
                scale_stage_layers.append(nn.Sequential(*layers))
            stages_sampling.append(nn.ModuleList(scale_stage_layers))
            """
            p3-> 4个特征图通道数缩小为1/2，特征图上采样放大2倍，然后沿通道数concat
            p4-> 保持原样不动
            p5-> 4个特征图通道数不变, 特征图下采样缩小为1/2，然后沿通道数concat
            p6-> 对results中的最后一个使用最大池化, 使用最大池化将特征图下采样缩小为1/2
            """
            # 特征图输入c2f模块时对应的通道数
            in_dim = int(
                sum([in_channel // max(1, scale) for in_channel in in_channels])
            )
            stage_layers: list[nn.Module] = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                cast(nn.Module, LayerNorm(out_channels)),
            ]
            stages.append(nn.Sequential(*stage_layers))
            stages_levels.append(lvl)
        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)
        self.stages_levels = stages_levels

    def forward(self, x: list[torch.Tensor]):
        logger.debug(
            "输入投影层特征层形状为：{}",
            {f"feature{i}": feature.shape for i, feature in enumerate(x)},
        )

        """
            p3-> 4个特征图通道数缩小为1/2，特征图上采样放大2倍，然后沿通道数concat
            p4-> 保持原样不动
            p5-> 4个特征图通道数不变, 特征图下采样缩小为1/2，然后沿通道数concat
            p6-> 将results中的最后一个使用最大池化，特征图下采样缩小为1/2
        """
        results = []
        for i, stage in enumerate(self.stages):
            feat_fuse_list = []
            # !在scale下对4个特征图做统一的缩放处理
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse_list.append(stage_sampling(x[j]))

            # !缩放4个特征图后沿通道数concat [B, 4*C, H, W]
            feat_fuse = (
                torch.cat(feat_fuse_list, dim=1)
                if len(feat_fuse_list) > 1
                else feat_fuse_list[0]
            )
            logger.debug(
                f"{self.stages_levels[i]}：4个特征层经过concat后输出形状为：{feat_fuse.shape}"
            )
            # !将concat后的特征图输入C2f模块
            results.append(stage(feat_fuse))
            logger.debug(
                f"{self.stages_levels[i]}：4个特征层经过c2f模块后输出形状为：{results[-1].shape}"
            )

        # !对results中的最后一个使用最大池化，特征图下采样缩小为1/2
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )
        return results
