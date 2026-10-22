import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw


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
        boxes: list[list[float]],
    ) -> dict[str, torch.Tensor]:
        """转换单张图片及其边界框。

        image：PIL 图片，转换为 RGB 后进行增强。
        boxes：像素 xyxy 坐标列表，每个框包含 4 个浮点数，无目标时传入 []。
        返回 image（归一化后的 CHW float32 张量）与 boxes（像素 xyxy，
        形状为 (N, 4) 的 float32 张量）。框的数量和顺序保持不变。
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
        return {
            "image": result["image"],
            # 固定 dtype 和形状，保证无目标样本也返回 (0, 4)。
            "boxes": torch.as_tensor(result["bboxes"], dtype=torch.float32).reshape(
                -1, 4
            ),
        }


def draw_boxes(preview: Image.Image, preview_boxes: torch.Tensor) -> Image.Image:
    """在图片副本上画框，避免修改测试输入。"""
    preview = preview.copy()
    draw = ImageDraw.Draw(preview)
    for box in preview_boxes.tolist():
        draw.rectangle(tuple(box), outline="red", width=3)
    return preview


if __name__ == "__main__":
    from pathlib import Path

    from datasets import load_from_disk

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

        # 归一化张量不能直接显示：先执行 pixel * std + mean，还原到 [0, 1]。
        restored = pixels * pixels.new_tensor(std)[:, None, None]
        restored = restored + pixels.new_tensor(mean)[:, None, None]
        restored = (
            restored.clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8)
        )
        preview = Image.fromarray(restored.cpu().numpy())
        draw_boxes(preview, transformed_boxes).save(output_dir / f"{mode}.png")
        print(
            f"{mode}: 图片 {tuple(pixels.shape)} / {pixels.dtype}，"
            f"边界框 {tuple(transformed_boxes.shape)} / {transformed_boxes.dtype}"
        )

    # 训练模式强制水平翻转，翻转后的框应与验证模式的框形成左右镜像。
    expected = results["val"]["boxes"].clone()
    width = results["val"]["image"].shape[-1]
    expected[:, [0, 2]] = width - expected[:, [2, 0]]
    torch.testing.assert_close(results["train"]["boxes"], expected, rtol=0, atol=1e-3)
    print(f"检查通过，原图、训练和验证的带框预览已保存到：{output_dir}")
