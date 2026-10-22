import torch.nn as nn
from pathlib import Path
from transformers import AutoBackbone


class DinoV3(nn.Module):
    def __init__(self, root: str):
        super().__init__()

        cache_dir = Path(root) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.backbone = AutoBackbone.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            cache_dir=cache_dir,
            out_indices=(2, 5, 8, 11),
        )

    def forward(self, x):
        return self.backbone(x)
