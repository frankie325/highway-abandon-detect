from typing import Any
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
        else:
            # Transformers 的 from_pretrained 默认可能返回 eval 模式；解冻后显式启用训练模式。
            self.encoder.train()

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
        raw_feats = self.encoder(tensor_list.tensors, tensor_list.mask)
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

    def get_named_param_lr_pairs(
        self,
        config: AppConfig,
        prefix: str = "backbone.0",  # prefix: 主干网在RAODDETR整个模型的参数名前缀
    ) -> dict[str, dict[str, Any]]:
        # 从实际 DINOv3 模型读取 Transformer 层数；out_indices 只表示提取哪些层。
        num_layers = len(self.encoder.dino.model.layer)
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            # 如果encoder参数没有冻结，则分层处理学习率
            if backbone_key in n and p.requires_grad:
                lr = (
                    config.train_config.lr_encoder
                    * get_lr_decay_rate(
                        n,
                        lr_decay_rate=config.train_config.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * config.train_config.lr_component_decay**2
                )
                wd = config.train_config.weight_decay * get_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_lr_decay_rate(
    name: str, lr_decay_rate: float = 1.0, num_layers: int = 12
) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: Parameter name.
        lr_decay_rate: Base lr decay rate.
        num_layers: Number of ViT blocks.

    Returns:
        Lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    # 浅层提取通用低级特征（边缘、纹理），预训练权重已经很好，用小学习率保护它不被破坏；
    # 深层更贴近下游检测任务，用大学习率适应新任务
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate


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
    position_embedding_module = build_position_encoding(config.model.d_model, "v2")

    backbone = Backbone(
        config=config,
        pretrain_model=config.model.pretrain_model,
        out_indices=config.model.out_indices,
        freeze_encoder=config.model.freeze_encoder,
        out_channels=config.model.d_model,
        projector_scale=config.model.projector_scale,
    )

    model = Joiner(backbone, position_embedding_module)
    return model
