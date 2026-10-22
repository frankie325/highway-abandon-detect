from typing import TypedDict
from PIL import Image
import torch


class RAODSample(TypedDict):
    """RAOD 原始单条样本的字段类型，适用于默认解码格式。

    仅提供静态类型提示，不做运行时校验；转换为张量后需使用对应的类型。
    """

    image: Image.Image  # 默认解码为 PIL 图片。
    image_id: str
    width: int  # 原始图片宽度，单位为像素。
    height: int  # 原始图片高度，单位为像素。
    boxes: list[list[float]]  # 每个框为 [xmin, ymin, xmax, ymax] 像素坐标。
    labels: list[int]  # 与 boxes 一一对应的类别编号。


class RAODBatch(TypedDict):
    """组装后的检测批次，所有列表均与图片顺序一一对应。"""

    images: torch.Tensor  # float32，[B, 3, H, W]，右下补边后的图片。
    image_ids: list[str]  # 长度为 B，保留原始字符串编号。
    origins: torch.Tensor  # int64，原始 [width, height]，形状为 [B, 2]。
    boxes: list[torch.Tensor]  # 每张图为 float32 [N_i, 4]，缩放后的像素 xyxy。
    labels: list[torch.Tensor]  # 每张图为 int64 [N_i]，与对应框顺序一致。
    padding_masks: torch.Tensor  # bool，[B, H, W]，True 为补边区域。
