"""使用 Hugging Face `datasets` 加载 RAOD-benchmark 数据集。

目录结构（data/RAOD-benchmark/）：
    spill_images_train_mosaic/   训练图片  (*.jpg)
    spill_images_test_mosaic/    测试图片  (*.jpg)
    spill_xmls_train_mosaic/     训练标注  (VOC *.xml)
    spill_xmls_test_mosaic/      测试标注  (VOC *.xml)
    select_1000_images/          1000 张子集图片
    select_1000_xmls/            子集标注  (VOC *.xml)
    select_1000_txts/            子集标注  (YOLO 归一化 *.txt)

标注说明：
    - 类别固定为 spill（抛洒 / 遗弃物），class_id = 0
    - xml 为绝对坐标 xyxy；txt 为 YOLO 归一化 cxcywh
    - "*_mosaic" 表示图像已做 mosaic 拼接增强，bbox 已是拼接后坐标

依赖：需先 `uv add datasets`。
"""

from __future__ import annotations
from functools import partial
from pathlib import Path
from collections.abc import Iterator
import xml.etree.ElementTree as ET
from datasets import DatasetDict, Dataset, Features, Value, Image, Sequence
from loguru import logger

CATEGORY = "spill"
CLASS_NAMES = [CATEGORY]
# 检测任务的 schema：bbox 用 [[xmin, ymin, xmax, ymax], ...] 存绝对坐标，
# 并保留图像宽高，便于后续缩放 / 转 YOLO 格式。
FEATURES = Features(
    {
        "image": Image(),
        "image_id": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "boxes": Sequence(Sequence(Value("float32"))),
        "labels": Sequence(Value("int64")),
    }
)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data" / "RAOD-benchmark"
SMALL_TRAIN = DATA_ROOT / "select_1000_images"
SMALL_LABELS = DATA_ROOT / "select_1000_xmls"
TRAIN = DATA_ROOT / "spill_images_train_mosaic"
TRAIN_LABELS = DATA_ROOT / "spill_xmls_train_mosaic"
TEST = DATA_ROOT / "spill_images_test_mosaic"
TEST_LABELS = DATA_ROOT / "spill_xmls_test_mosaic"
CACHE_DIR = ROOT / "cache"
RAOD = CACHE_DIR / "data" / "raod"
RAOD_SMALL = RAOD / "small"


def require_text(parent: ET.Element, tag: str) -> str:
    """断言子元素存在且文本非空，返回其文本（数据规范、无脏数据）。"""
    el = parent.find(tag)
    assert el is not None, f"缺少 <{tag}> 节点"
    assert el.text is not None, f"<{tag}> 文本为空"
    return el.text


def parse_xml(xml_path: Path) -> tuple[list[list[float]], list[int], int, int]:
    """解析 VOC 风格 xml，返回 (boxes, labels, width, height)。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    assert size is not None, f"{xml_path}: 缺少 <size> 节点"
    w = int(require_text(size, "width"))
    h = int(require_text(size, "height"))
    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        assert b is not None, f"{xml_path}: <object> 缺少 <bndbox>"
        boxes.append(
            [
                float(require_text(b, "xmin")),
                float(require_text(b, "ymin")),
                float(require_text(b, "xmax")),
                float(require_text(b, "ymax")),
            ]
        )
        name = require_text(obj, "name")
        labels.append(CLASS_NAMES.index(name) if name in CLASS_NAMES else -1)
    return boxes, labels, w, h


def gen(
    images_dir: Path,
    xmls_dir: Path,
) -> Iterator[dict[str, str | int | list[list[float]] | list[int]]]:
    # 获取文件下所有图片的路径
    image_paths = images_dir.glob("*.jpg")
    for img_path in image_paths:
        logger.debug("解析图片：{}", img_path)
        xml_path = xmls_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            logger.warning("标注不存在，跳过图片：{}", img_path)
            continue
        boxes, labels, w, h = parse_xml(xml_path)
        yield {
            "image": str(img_path),
            "image_id": img_path.stem,
            "width": w,
            "height": h,
            "boxes": boxes,
            "labels": labels,
        }


def process():
    logger.info("开始生成 RAOD 缓存，输出目录：{}", RAOD)
    # ① cache_dir：指定 from_generator 的【生成过程临时缓存】位置。
    #    from_generator 是即时执行的，调用时会把整批数据物化成 Arrow 并临时写进该目录
    #    （缓存文件名由参数 hash 得到，如 cache/generator-xxxx/xxx.arrow）。
    #    作用：再次以相同参数调用 from_generator 时命中缓存、跳过重新解析 xml。
    #    注意：这只是"过程缓存"，是 hash 命名的临时文件，不是可直接交付的成品数据集。
    ds = Dataset.from_generator(
        partial(gen, SMALL_TRAIN, SMALL_LABELS),
        features=FEATURES,
        cache_dir=str(CACHE_DIR),
    )
    # 清理缓存的文件
    # print(ds.cleanup_cache_files())

    # ② save_to_disk：把生成好的 Dataset【正式持久化】到语义化路径。
    #    落盘为可直接使用的成品数据集（含 dataset_info.json 元数据 + 分片 .arrow），
    #    下游可用 load_from_disk("cache/data/raod") 稳定加载。
    #    与 ① 的区别：① 是 hash 命名的临时缓存（仅用于加速重生成），
    #    ② 是路径清晰、稳定的成品数据集。
    #    两者内容相同，但都写在 cache/ 下 => 同一份数据被存了两份，占用双倍空间。
    ds.save_to_disk(RAOD_SMALL)
    logger.success("小样本缓存已保存：{} 条，路径：{}", len(ds), RAOD_SMALL)

    train_ds = Dataset.from_generator(
        partial(gen, TRAIN, TRAIN_LABELS),
        features=FEATURES,
        cache_dir=str(CACHE_DIR),
    )

    test_ds = Dataset.from_generator(
        partial(gen, TEST, TEST_LABELS),
        features=FEATURES,
        cache_dir=str(CACHE_DIR),
    )

    dict = DatasetDict({"train": train_ds, "test": test_ds})
    dict.save_to_disk(RAOD, max_shard_size=500)  # 数据集分片的最大大小: 500MB
    logger.success("RAOD 缓存已保存：训练 {} 条，测试 {} 条", len(train_ds), len(test_ds))


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    from src.config import load_config

    setup_logging(load_config().logger)
    with logger.catch(reraise=True):
        process()
    # print(SMALL_TRAIN)
    # data_gen = gen(SMALL_TRAIN, SMALL_LABELS)
    # print(next(data_gen))
