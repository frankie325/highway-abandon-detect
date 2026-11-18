from .backbone.dinov3 import DinoV3
from .backbone.backbone import build_backbone
from .haoddetr import HAODDETR
from .transformer import build_transformer
from src.config import AppConfig


def build_model(config: AppConfig) -> HAODDETR:
    backbone = build_backbone(config)
    transformer = build_transformer(config)
    model = HAODDETR(
        backbone,
        transformer,
        num_classes=config.model.num_classes,
        num_queries=config.model.num_queries,
        aux_loss=config.model.aux_loss,
        group_detr=config.model.group_detr,
        two_stage=config.model.two_stage,
        bbox_reparam=config.model.bbox_reparam,
    )

    return model


__all__ = ["DinoV3"]
