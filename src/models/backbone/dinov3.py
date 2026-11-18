import torch.nn as nn
from transformers import AutoBackbone
from loguru import logger
from typing import cast
import torch
import torch.nn.functional as F


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

    def forward(
        self, images: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        # images 为 B, C, H, W；没有 padding_mask 时不屏蔽任何 token。
        attention_mask = None
        if padding_mask is not None:
            # 像素级 mask 为 B, H, W：True 表示补边，False 表示有效像素。
            if padding_mask.shape != (images.shape[0], images.shape[2], images.shape[3]):
                raise ValueError("padding_mask 必须与图片的 batch、高和宽一致")
            patch_size = self.dino.config.patch_size
            # 每个 patch 内的布尔值转为 0/1 后求平均；仅平均值为 1 时，
            # 使用平均池化检查一个 patch 内的所有像素，只有全部为 padding 才屏蔽，避免把图像边缘上的目标信息一起屏蔽掉
            # B, H, W -> B, 1, H, W -> B, H/P * W/P，P 为 patch_size。
            padded_patches = F.avg_pool2d(
                padding_mask.to(device=images.device, dtype=torch.float32).unsqueeze(1),
                kernel_size=patch_size,
                stride=patch_size,
            ).flatten(1) == 1
            # DINOv3 的序列还包含 1 个 CLS token 和若干 register token；
            # 在 patch token 前补 False，使这些特殊 token 始终可以被注意力访问。
            num_prefix = 1 + getattr(self.dino.config, "num_register_tokens", 0)
            prefix = torch.zeros(
                images.shape[0], num_prefix, dtype=torch.bool, device=images.device
            )
            padded_tokens = torch.cat((prefix, padded_patches), dim=1)
            # 将布尔 mask 转成注意力分数使用的加性 mask：有效位置为 0，
            # 补边位置为 -inf；分数加上 -inf 后，softmax 对应权重为 0。
            attention_mask = torch.zeros_like(padded_tokens, dtype=torch.float32)
            attention_mask.masked_fill_(padded_tokens, float("-inf"))
            # B, token数 -> B, 1, 1, token数，广播到所有注意力头和 query。
            attention_mask = attention_mask[:, None, None, :]
        # mask 随 backbone 传入 DINOv3 每层自注意力，屏蔽补边 key/value。
        outputs = self.dino(images, attention_mask=attention_mask)
        feature_maps = cast(
            tuple[torch.Tensor], (outputs.feature_maps)
        )  # 总共4层特征图，[(batch_size, dim , h ,w), ...]

        return feature_maps
