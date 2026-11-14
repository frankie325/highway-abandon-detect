import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
from loguru import logger

from src.utils.box_ops import box_xyxy_to_cxcywh


class AlbTransform:
    """同步转换 RGB 图片和像素坐标边界框，不做图像裁剪或 padding。

    训练：等比例缩放、随机水平翻转、随机亮度/对比度增强、归一化。
    验证：等比例缩放、归一化，不应用随机增强。
    scale_size 指缩放后的最长边；不同长宽比的图片仍可能具有不同尺寸。
    mean/std 默认使用 ImageNet 参数，可根据预训练权重调整。
    """

    def __init__(
        self,
        train: bool = True,
        *,
        scale_size: int = 1280,
        flip_p: float = 0.5,
        color_p: float = 0.3,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        seed: int | None = None,
    ) -> None:
        self.transform = A.Compose(
            [
                # 保持长宽比，将最长边缩放至 scale_size，并同步缩放边界框。
                A.LongestMaxSize(max_size=scale_size, p=1.0),
                # 几何增强必须与边界框同步；验证阶段关闭随机翻转。
                A.HorizontalFlip(p=flip_p if train else 0.0),
                # 颜色增强只改变像素，不改变框的位置；验证阶段关闭。
                A.RandomBrightnessContrast(
                    brightness_range=(-0.1, 0.1),
                    contrast_range=(-0.1, 0.1),
                    p=color_p if train else 0.0,
                ),
                # 将 uint8 像素按 mean/std 归一化，再转为 CHW 张量。
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                # pascal_voc 对应绝对像素坐标 [xmin, ymin, xmax, ymax]。
                coord_format="pascal_voc",
                # 将轻微越界的标注限制到图像范围，退化框仍会报错。
                clip_bboxes_on_input=True,
            ),
            seed=seed,
            strict=True,
            telemetry=False,
        )

    def set_random_seed(self, seed: int) -> None:
        """设置随机种子，使图片和边界框的联合增强可复现。"""
        self.transform.set_random_seed(seed)

    def __call__(
        self,
        image: Image.Image,
        boxes: list[tuple[float, float, float, float]],
    ) -> dict[str, torch.Tensor]:
        """转换单张图片及其边界框。

        image：PIL 图片，转换为 RGB 后进行增强。
        boxes：像素 xyxy 坐标列表，每个框包含 4 个浮点数，无目标时传入 []。
        返回 image（按 mean/std 归一化后的 CHW float32 张量）与 boxes
        （归一化到 [0, 1] 的 cxcywh 格式，形状为 (N, 4) 的 float32 张量）。
        框的数量和顺序保持不变。
        """
        # Albumentations 接收 NumPy 图片，将 PIL 图片转换为 HWC RGB 数组。
        # 直接在此确保连续（C-contiguous），满足 Albumentations/cv2 对图像内存布局的要求；
        # 已连续时 np.ascontiguousarray 不拷贝、零开销。
        image_array = np.ascontiguousarray(image.convert("RGB"))
        boxes_array = np.asarray(boxes, dtype=np.float32)
        # 空列表转换后是一维数组，显式整理为 (0, 4)。
        if boxes_array.shape == (0,):
            boxes_array = boxes_array.reshape(0, 4)
        if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
            raise ValueError("边界框必须为 (N, 4) 格式的像素 xyxy 坐标")
        if not np.isfinite(boxes_array).all():
            raise ValueError("边界框坐标必须为有限数值")

        # 使用关键字传参，让同一次随机变换同时作用于图片和边界框。
        result = self.transform(image=image_array, bboxes=boxes_array)
        image_out = result["image"]  # ToTensorV2 输出 CHW 张量
        boxes = result["bboxes"]
        # 用变换后图片的真实尺寸做归一化基准，因为 LongestMaxSize 已把图片
        # 与边界框缩放到新坐标系；若仍用原始尺寸会导致归一化值错误。
        height, width = int(image_out.shape[-2]), int(image_out.shape[-1])
        boxes_ts = torch.as_tensor(boxes, dtype=torch.float32)
        boxes_ts = box_xyxy_to_cxcywh(boxes_ts)  # 转换为cx,cy,w,h    N, 4

        # 归一化
        boxes_ts = boxes_ts / torch.as_tensor(
            [width, height, width, height], dtype=torch.float32
        )
        return {
            "image": image_out,
            # 固定 dtype 和形状，保证无目标样本也返回 (0, 4)。
            "boxes": boxes_ts,
        }


def draw_boxes(preview: Image.Image, preview_boxes: torch.Tensor) -> Image.Image:
    """在图片副本上画框，避免修改测试输入。"""
    preview = preview.copy()
    draw = ImageDraw.Draw(preview)
    for box in preview_boxes.tolist():
        draw.rectangle(tuple(box), outline="red", width=3)
    return preview


def tensor_to_image(
    image: torch.Tensor,
    boxes: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    highlight_padding: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """将归一化的 [3, H, W] 张量还原为 RGB 图片，并绘制像素 xyxy 框。

    boxes 为 [N, 4]，无目标时传入形状为 [0, 4] 的张量。
    padding_mask 为 bool [H, W]，True 表示补边；传入时默认用半透明紫色标记。
    highlight_padding=False 可查看补边的真实显示颜色。始终保留完整画布，不裁掉补边。
    mean/std 必须与图像转换时一致；反归一化不会撤销缩放、翻转或颜色增强。
    支持设备上的张量，内部 detach 并移到 CPU，不修改输入数据。
    """
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("image 必须为归一化后的浮点张量，形状为 [3, H, W]")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes 必须为 [N, 4] 格式的像素 xyxy 张量")
    pixels = image.detach().to(device="cpu", dtype=torch.float32)
    preview_boxes = boxes.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(pixels).all() or not torch.isfinite(preview_boxes).all():
        raise ValueError("图片和边界框不能包含 NaN 或无穷值")

    # Albumentations 的归一化为 (pixel / 255 - mean) / std，这里执行逆操作。
    restored = pixels * pixels.new_tensor(std)[:, None, None]
    restored = restored + pixels.new_tensor(mean)[:, None, None]
    restored = restored.clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8)
    preview = Image.fromarray(restored.numpy())

    if padding_mask is not None:
        if padding_mask.dtype != torch.bool or tuple(padding_mask.shape) != tuple(
            pixels.shape[-2:]
        ):
            raise ValueError("padding_mask 必须为与图片宽高一致的 bool [H, W] 张量")
        if highlight_padding:
            # 紫色仅用于可视化，不代表模型收到的实际像素；有效区域保持不变。
            alpha = padding_mask.detach().cpu().to(torch.uint8).mul(96).numpy()
            overlay = Image.new("RGBA", preview.size, (255, 0, 255, 0))
            overlay.putalpha(Image.fromarray(alpha))
            preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert(
                "RGB"
            )

    # 补边位于右下，不改变框的原点；按转换后的坐标直接绘制即可。
    return draw_boxes(preview, preview_boxes)


def batch_tensor_to_images(
    images: torch.Tensor,
    boxes: list[torch.Tensor] | torch.Tensor,
    *,
    padding_masks: torch.Tensor | None = None,
    highlight_padding: bool = True,
    concat: bool = False,
    columns: int = 2,
    gap: int = 8,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> list[Image.Image] | Image.Image:
    """还原 [B, 3, H, W] 批次，默认返回 B 张带框 PIL 图片。

    boxes 接收 RAODBatch 的变长列表（每张 [N_i, 4]），或固定框数的 [B, N, 4] 张量。
    padding_masks 为 bool [B, H, W]；补边不会被裁掉，标记方式与单图方法一致。
    concat=True 时返回一张按 columns 列拼接的图片，包含样本索引和补边占比。
    拼图间隔为 gap 像素；拼图不再次缩放图片，边界框位置保持一致。
    """
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
        raise ValueError("images 必须为非空的 [B, 3, H, W] 批次张量")
    batch_size, _, height, width = images.shape
    if isinstance(boxes, torch.Tensor) and (boxes.ndim != 3 or boxes.shape[-1] != 4):
        raise ValueError("批量 boxes 张量必须为 [B, N, 4]，变长框请使用张量列表")
    if len(boxes) != batch_size:
        raise ValueError("boxes 的图片数量必须与 images 一致")
    if padding_masks is not None:
        if padding_masks.dtype != torch.bool or tuple(padding_masks.shape) != (
            batch_size,
            height,
            width,
        ):
            raise ValueError("padding_masks 必须为与批次尺寸一致的 bool [B, H, W] 张量")

    previews = [
        tensor_to_image(
            image,
            boxes[index],
            padding_mask=padding_masks[index] if padding_masks is not None else None,
            highlight_padding=highlight_padding,
            mean=mean,
            std=std,
        )
        for index, image in enumerate(images)
    ]
    if not concat:
        return previews
    if columns < 1 or gap < 0:
        raise ValueError("columns 必须为正整数，gap 不能为负数")

    columns = min(columns, batch_size)
    rows = (batch_size + columns - 1) // columns
    header_height, label_height = 24, 22
    grid = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * gap,
            header_height + rows * (height + label_height) + (rows + 1) * gap,
        ),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(grid)
    marked = highlight_padding and padding_masks is not None
    legend = (
        "red: boxes | magenta: padding overlay"
        if marked
        else "red: boxes | padding: original fill color"
    )
    draw.text((gap, 4), legend, fill="white")
    for index, preview in enumerate(previews):
        x = gap + (index % columns) * (width + gap)
        y = header_height + gap + (index // columns) * (height + label_height + gap)
        label = f"sample {index}"
        if padding_masks is not None:
            percentage = padding_masks[index].detach().float().mean().item() * 100
            label += f" | padding {percentage:.1f}%"
        draw.text((x, y + 3), label, fill="white")
        grid.paste(preview, (x, y + label_height))
    return grid


if __name__ == "__main__":
    from pathlib import Path

    from datasets import load_from_disk
    from src.utils.logger import setup_logging
    from src.config import load_config

    config = load_config()
    setup_logging(config.logger)

    # 使用同一条真实 RAOD 样本，对比训练和验证模式；可修改索引查看其他图片。
    root = Path(__file__).resolve().parents[2]
    dataset = load_from_disk(root / "cache" / "data" / "raod" / "small")
    sample = dataset[0]
    image = sample["image"].convert("RGB")
    boxes = sample["boxes"]
    scale_size = 640
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    output_dir = root / "cache" / "transform_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_boxes(image, torch.tensor(boxes).reshape(-1, 4)).save(
        output_dir / "original.png"
    )
    results = {}
    for mode, is_train in (("train", True), ("val", False)):
        # 仅在此测试中把增强概率设为 1，方便观察翻转及颜色变化。
        # 验证模式即使传入相同概率，也应关闭这两项随机增强。
        transform = AlbTransform(
            train=is_train,
            scale_size=scale_size,
            flip_p=1.0,
            color_p=1.0,
            mean=mean,
            std=std,
            seed=42,
        )
        result = transform(image, boxes)
        results[mode] = result
        pixels, transformed_boxes = result["image"], result["boxes"]

        # 检查输出类型、形状以及数值，确保框数量没有改变。
        assert pixels.dtype == torch.float32 and pixels.shape[0] == 3
        assert max(pixels.shape[-2:]) == scale_size
        assert transformed_boxes.dtype == torch.float32
        assert transformed_boxes.shape == (len(boxes), 4)
        assert torch.isfinite(pixels).all() and torch.isfinite(transformed_boxes).all()
        assert transform(image, [])["boxes"].shape == (0, 4)

        if not is_train:
            # 验证模式重复处理同一输入，图片和框应完全一致。
            repeated = transform(image, boxes)
            for key in result:
                torch.testing.assert_close(result[key], repeated[key], rtol=0, atol=0)

        tensor_to_image(pixels, transformed_boxes, mean=mean, std=std).save(
            output_dir / f"{mode}.png"
        )
        logger.info(
            "{}：图片 {} / {}，边界框 {} / {}",
            mode,
            tuple(pixels.shape),
            pixels.dtype,
            tuple(transformed_boxes.shape),
            transformed_boxes.dtype,
        )

    # 训练模式强制水平翻转，翻转后的框应与验证模式的框形成左右镜像。
    expected = results["val"]["boxes"].clone()
    width = results["val"]["image"].shape[-1]
    expected[:, [0, 2]] = width - expected[:, [2, 0]]
    torch.testing.assert_close(results["train"]["boxes"], expected, rtol=0, atol=1e-3)

    from src.datasets.collate import RAODCollate

    # 选择宽高比差异最大的两张图片，再补足四张，确保能直观看到批次补边。
    ratios = [
        row["width"] / row["height"]
        for row in dataset.select_columns(["width", "height"])
    ]
    indices = list(
        dict.fromkeys((ratios.index(max(ratios)), ratios.index(min(ratios))))
    )
    for index in range(len(dataset)):
        if len(indices) >= 4:
            break
        if index not in indices:
            indices.append(index)
    samples = [dataset[index] for index in indices]
    for mode, is_train in (("train", True), ("val", False)):
        collate = RAODCollate(
            train=is_train,
            multi_scale=False,
            fix_scale_size=scale_size,
            patch_size=config.model.patch_size,
            seed=config.seed,
        )
        batch = collate(samples)
        # 原样拼图保留反归一化后的真实补边颜色，标记版叠加紫色显示补边范围。
        for highlight in (False, True):
            grid = batch_tensor_to_images(
                batch["images"],
                batch["boxes"],
                padding_masks=batch["padding_masks"],
                highlight_padding=highlight,
                concat=True,
                columns=2,
                mean=mean,
                std=std,
            )
            assert isinstance(grid, Image.Image)
            suffix = "_padding" if highlight else ""
            grid.save(output_dir / f"batch_{mode}{suffix}.png")
        logger.info(
            "{} 批次预览：images={}，原始样本索引={}",
            mode,
            tuple(batch["images"].shape),
            indices,
        )
    logger.success("检查通过，原图、训练和验证的带框预览已保存到：{}", output_dir)
