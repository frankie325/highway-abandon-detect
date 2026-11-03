import torch.nn as nn
from transformers import AutoBackbone
from loguru import logger
from typing import cast
import torch


class DinoV3(nn.Module):
    def __init__(
        self,
        pretrain_model: str,
        cache_dir: str,
        out_indices: list[int] = [3, 6, 9, 12],
    ):
        super().__init__()

        logger.info("加载Dinov3预训练权重...")
        self.dino = AutoBackbone.from_pretrained(
            pretrain_model,
            # out_features=[f"stage{i}" for i in out_indices],
            cache_dir=cache_dir,
            out_indices=out_indices,
        )

        logger.info("加载Dinov3预训练权重成功")

        self.out_feature_channels = [
            self.dino.out_feature_channels[f"stage{i}"] for i in out_indices
        ]  # 例如 DINOv3 vitb16: [768, 768, 768, 768]
        logger.debug(
            "Dinov3输出特征层的通道数：{}",
            {
                f"state{value}": self.out_feature_channels[index]
                for index, value in enumerate(out_indices)
            },
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        outputs = self.dino(images)
        feature_maps = cast(
            tuple[torch.Tensor], (outputs.feature_maps)
        )  # 总共4层特征图，[(batch_size, dim , h ,w), ...]

        return feature_maps
