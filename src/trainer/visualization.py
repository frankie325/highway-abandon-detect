"""绘制验证预览与原图上的预测结果。"""

from collections.abc import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

from src.datasets.raod_benchmark import CLASS_NAMES
from src.datasets.transforms import tensor_to_image
from src.utils.box_ops import box_cxcywh_to_xyxy
from src.utils.tensors import NestedTensor


def draw_prediction_boxes(
    image: Image.Image,
    prediction: dict[str, torch.Tensor],
    resized_size: tuple[int, int],
    *,
    class_names: Sequence[str],
    score_threshold: float,
) -> int:
    """将缩放后的像素框映射回原图，原地绘制并返回绘制数量。"""
    keep = prediction["scores"] >= score_threshold
    if not keep.any():
        return 0

    resized_width, resized_height = resized_size
    scale_x = image.width / resized_width
    scale_y = image.height / resized_height
    boxes = prediction["boxes"][keep] * prediction["boxes"].new_tensor(
        [scale_x, scale_y, scale_x, scale_y]
    )
    scores = prediction["scores"][keep].tolist()
    labels = prediction["labels"][keep].tolist()

    draw = ImageDraw.Draw(image)
    font_size = max(12, min(image.size) // 50)
    font = ImageFont.load_default(size=font_size)
    for box, score, label in zip(boxes.tolist(), scores, labels, strict=True):
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, image.width - 1), min(x2, image.width - 1)
        y1, y2 = min(y1, image.height - 1), min(y2, image.height - 1)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=max(1, font_size // 6),
        )
        text = f"{class_names[label]} {score:.2f}"
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font, anchor="lt")
        text_x = min(x1, max(0, image.width - text_width - 1))
        text_y = min(max(0, y1 - text_height - 2), max(0, image.height - text_height - 1))
        draw.text(
            (text_x, text_y), text, fill="white", font=font,
            anchor="lt", stroke_width=1, stroke_fill="black",
        )
    return len(scores)


def detection_previews(
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    outputs: dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.3,
    max_predictions: int = 20,
) -> list[Image.Image]:
    """绘制绿色真实框、带类别名称和置信度的红色预测框及紫色 padding。"""
    assert samples.mask is not None
    previews = []
    for index, target in enumerate(targets):
        mask = samples.mask[index].detach().cpu()
        height = int((~mask[:, 0]).sum())
        width = int((~mask[0, :]).sum())
        if not height or not width:
            raise ValueError("验证图片没有有效像素区域")
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        preview = tensor_to_image(
            samples.tensors[index], torch.empty((0, 4)), padding_mask=mask
        )
        draw = ImageDraw.Draw(preview)

        def pixel_boxes(boxes: torch.Tensor) -> torch.Tensor:
            xyxy = box_cxcywh_to_xyxy(boxes.detach().cpu().float()) * scale
            xyxy[:, 0::2].clamp_(0, width)
            xyxy[:, 1::2].clamp_(0, height)
            return xyxy

        for box in pixel_boxes(target["boxes"]).tolist():
            draw.rectangle(tuple(box), outline="lime", width=3)

        probabilities = outputs["pred_logits"][index].detach().float().sigmoid().cpu()
        flat = probabilities.flatten()
        scores, selected = flat.topk(min(max_predictions, flat.numel()))
        selected = selected[scores >= score_threshold]
        scores = scores[scores >= score_threshold]
        queries = torch.div(selected, probabilities.shape[-1], rounding_mode="floor")
        classes = selected % probabilities.shape[-1]
        boxes = pixel_boxes(outputs["pred_boxes"][index][queries.to(outputs["pred_boxes"].device)])
        for box, score, label in zip(boxes.tolist(), scores.tolist(), classes.tolist(), strict=True):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            draw.rectangle(tuple(box), outline="red", width=2)
            draw.text(
                (box[0], max(0, box[1] - 12)),
                f"{CLASS_NAMES[label]}: {score:.2f}",
                fill="red",
                stroke_width=1,
                stroke_fill="white",
            )
        previews.append(preview)
    return previews
