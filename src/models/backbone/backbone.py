import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from loguru import logger

from .dinov3 import DinoV3
from .projector import MultiScaleProjector
from src.config import AppConfig
from src.utils.tensors import NestedTensor
from src.models.position_encoding import build_position_encoding


class Backbone(nn.Module):

    def __init__(
        self,
        config: AppConfig,
        pretrain_model: str,
        out_indices: list[int],
        freeze_encoder: bool = True,
        out_channels: int = 256,
        projector_scale: list[Literal["P3", "P4", "P5", "P6"]] = ["P3", "P4", "P5"],
        layer_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()
        cache_dir = config.root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = DinoV3(pretrain_model, str(cache_dir), out_indices)
        self.projector = MultiScaleProjector(
            projector_scale,
            self.encoder.out_feature_channels,
            out_channels,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def level_mask(
        feature_map: torch.Tensor,  # 特征图 (B, C , H/16 ,W/16)
        tensor_list: NestedTensor,
    ):
        """
        tensor_list:
            padding_masks: torch.Tensor,  # (B, H, W)
            no_padding: bool,  # 表示这批图像没有任何一张需要被 padding
        """
        b, _, h, w = feature_map.shape
        if tensor_list.no_padding:
            return torch.zeros((b, h, w), dtype=torch.bool, device=feature_map.device)
        assert tensor_list.mask is not None
        m: torch.Tensor = tensor_list.mask[
            None
        ].float()  # 按照interpolate输入要求，添加一个维度 (1, B, H, W)，并转为浮点数进行运算
        return F.interpolate(m, (h, w)).to(torch.bool)[0]

    def forward(
        self,
        tensor_list: NestedTensor,
    ) -> list[NestedTensor]:
        # tensor_list.tensors: (B, C, H, W)
        raw_feats = self.encoder(tensor_list.tensors)
        feats = self.projector(raw_feats)

        no_padding = tensor_list.no_padding
        outs: list[NestedTensor] = []
        # 通过下采样为不同尺寸的特征图生成mask
        for feat in feats:
            outs.append(
                NestedTensor(
                    feat,
                    self.level_mask(
                        feat, tensor_list
                    ),  # 将padding_mask下采样值特征图的大小
                    no_padding,
                ),
            )

        for i, out in enumerate(outs):
            logger.debug(
                "骨干网输出特征图{}的形状：{}，输出padding_mask形状：{}，这批图像没有任何一张需要被 padding：{}",
                i,
                out.tensors.shape,
                out.mask.shape,
                "是" if out.no_padding else "否",
            )
        return outs


class Joiner(nn.Sequential):
    def __init__(self, backbone: Backbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], list[torch.Tensor], list[NestedTensor] | None]:
        """"""
        # 经过骨干网
        result = self[0](tensor_list)
        if isinstance(result, tuple):
            x, cross_attn_x = result
        else:
            x, cross_attn_x = result, None
        pos = []
        # 得到特征图的位置编码
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos, cross_attn_x


def build_backbone(config: AppConfig):
    position_embedding_module = build_position_encoding(
        config.model.d_model, "v2"
    )

    backbone = Backbone(
        config=config,
        pretrain_model=config.model.pretrain_model,
        out_indices=config.model.out_indices,
        freeze_encoder=True,
        out_channels=config.model.d_model,
        projector_scale=config.model.projector_scale,
    )

    model = Joiner(backbone, position_embedding_module)
    return model
