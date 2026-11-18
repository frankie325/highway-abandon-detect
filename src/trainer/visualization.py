"""验证图片的边界框与 padding 预览。"""

import torch
from PIL import Image, ImageDraw

from src.datasets.transforms import tensor_to_image
from src.utils.box_ops import box_cxcywh_to_xyxy
from src.utils.tensors import NestedTensor


def detection_previews(
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    outputs: dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.3,
    max_predictions: int = 20,
) -> list[Image.Image]:
    """绘制绿色真实框、红色预测框和紫色 padding，保留完整画布。"""
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
            draw.text((box[0], max(0, box[1] - 12)), f"{label}: {score:.2f}", fill="red", stroke_width=1, stroke_fill="white")
        previews.append(preview)
    return previews
