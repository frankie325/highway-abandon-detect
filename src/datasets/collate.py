from collections.abc import Sequence
import torch

from .transforms import AlbTransform
from .types import RAODSample, RAODBatch
from src.utils.tensors import NestedTensor


class RAODCollate:
    """同步转换样本并补边，按 RAODBatch 的字段组装模型输入。

    同一批共享缩放长边；验证或关闭多尺度时使用 fix_scale_size。
    图片和框由 AlbTransform 同步转换，框和类别按图片保留为变长列表。
    """

    def __init__(
        self,
        train: bool = True,
        *,
        multi_scale: bool = True,
        scale_sizes: Sequence[int] = (1024, 1152, 1280),
        fix_scale_size: int = 1280,
        patch_size: int = 16,
        seed: int = 42,
    ) -> None:
        if fix_scale_size < 1 or patch_size < 1:
            raise ValueError("fix_scale_size 和 patch_size 必须为正整数")
        if not 0 <= seed < 2**32:
            raise ValueError("seed 必须处于 [0, 2**32) 范围")
        self.train = train
        self.multi_scale = multi_scale
        self.scale_sizes = tuple(dict.fromkeys(scale_sizes))
        self.fix_scale_size = fix_scale_size
        self.patch_size = patch_size
        self.seed = seed

        sizes = self.scale_sizes if train and multi_scale else (fix_scale_size,)
        # 每个sacle尺度都进行复用
        self.transforms = {
            size: AlbTransform(train=train, scale_size=size, seed=seed)
            for size in sizes
        }

    def __call__(
        self, batch: list[RAODSample]
    ) -> tuple[NestedTensor, list[dict[str, torch.Tensor]]]:
        if not batch:
            raise ValueError("batch 不能为空")
        if any(len(sample["boxes"]) != len(sample["labels"]) for sample in batch):
            raise ValueError("每张图片的 boxes 和 labels 必须一一对应")

        transform = self.transforms[self.get_scale_size()]
        results = [transform(sample["image"], sample["boxes"]) for sample in batch]
        # 单张图片为 [C, H, W]。等比例缩放只统一最长边，另一条边仍可能不同，
        # 因此分别取批次最大高、最大宽，确定能容纳所有图片的公共画布。
        images = [result["image"] for result in results]
        max_height = max(image.shape[-2] for image in images)
        max_width = max(image.shape[-1] for image in images)

        # 向上取整到配置中 patch_size 的整数倍，满足骨干网络的尺寸对齐要求。
        # 公式等价于 ceil(尺寸 / patch_size) * patch_size；已整除时不变。
        # 例如 patch_size=16 时，721 补到 736，720 保持不变。
        # 高、宽分别计算，不要求画布为正方形；ViT patch_size=16 时可按 16 对齐。
        padded_height = (
            (max_height + self.patch_size - 1) // self.patch_size * self.patch_size
        )
        padded_width = (
            (max_width + self.patch_size - 1) // self.patch_size * self.patch_size
        )

        # 创建 [B, 3, H_pad, W_pad] 的零填充画布，沿用图片的 dtype 和 device。
        # 此处的 0 是归一化后的数值，不一定对应原始图片中的黑色。
        batch_szie = len(images)
        device = images[0].device

        # 如果这批图像的尺寸和padding过后的尺寸一样，说明这批图像没有任何一张需要被 padding
        no_padding = all(
            img.shape[-2:] == (padded_height, padded_width) for img in images
        )

        if no_padding:
            padding_mask = torch.zeros(
                (batch_szie, padded_height, padded_width),
                dtype=torch.bool,
                device=device,
            )
            pixel_values = torch.stack(images, dim=0)
        else:
            pixel_values = images[0].new_zeros(
                (batch_szie, 3, padded_height, padded_width)
            )
            # 每张图片对应一张 [H_pad, W_pad] 的 bool 掩码，无需通道维度。
            # 先将所有位置标为 True（padding），再把图片实际覆盖的位置标为 False。
            padding_mask = torch.ones(
                (len(images), padded_height, padded_width),
                dtype=torch.bool,
                device=device,
            )
            for i, image in enumerate(images):
                height, width = image.shape[-2:]
                # 图片放在画布左上角，仅右侧和底部补边，图片内容不会再次缩放或拉伸。
                # 左上角坐标不变，所以补边无需平移边界框。
                pixel_values[i, :, :height, :width] = image
                # 按图片尺寸标记有效区域，与像素值无关：原图中的黑色像素同样有效。
                padding_mask[i, :height, :width] = False

        targets = [
            {
                "labels": torch.tensor(sample["labels"], dtype=torch.int64),
                "boxes": results[i]["boxes"],
            }
            for i, sample in enumerate(batch)
        ]  # list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]

        tensors = NestedTensor(
            pixel_values, padding_mask, no_padding  # (B, C, H, W)  # (B, H, W)
        )
        return tensors, targets
        # 检测框和类别数量可能不同，按图片保留变长列表，不强行堆叠。
        # return {
        #     "images": pixel_values,  # (B, C, H, W)
        #     "image_ids": [sample["image_id"] for sample in batch],
        #     "no_padding": no_padding,
        #     "origins": torch.tensor(
        #         [[sample["width"], sample["height"]] for sample in batch],
        #         dtype=torch.int64,
        #     ),
        #     "boxes": [result["boxes"] for result in results],  # list[N, 4), ...]
        #     "labels": [
        #         torch.tensor(sample["labels"], dtype=torch.int64) for sample in batch
        #     ],  # list[(N,) ,...]
        #     "padding_masks": padding_mask,  # (B, H, W)
        # }

    def get_scale_size(self) -> int:
        """训练时随机选择一个缩放尺寸，非训练时返回固定尺寸。"""
        if not self.train or not self.multi_scale:
            return self.fix_scale_size

        # 使用 PyTorch 随机源，由训练入口和 DataLoader worker 的种子控制。
        index = torch.randint(len(self.scale_sizes), ()).item()
        return self.scale_sizes[index]
