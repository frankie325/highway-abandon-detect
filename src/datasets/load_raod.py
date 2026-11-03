from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader
from loguru import logger

from .collate import RAODCollate
from src.config import TrainConfig

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


# 小型数据集，用于快速验证
def load_raod_small(train: bool = True) -> Dataset:
    logger.info("加载raod_small数据集...")
    text = "训练集" if train else "测试集"
    ds = load_from_disk(RAOD_SMALL)
    small_dict: DatasetDict = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    dataset = small_dict["train" if train else "test"]
    logger.info("{}加载完成：共{} 张图片", text, dataset.num_rows)
    return dataset


def build_dataloader(
    train: bool, config: TrainConfig, *, patch_size: int
) -> DataLoader:
    dataset: Dataset = load_raod_small(train)
    collator = RAODCollate(
        train=train,
        multi_scale=config.multi_scale,
        scale_sizes=config.scale_sizes,
        fix_scale_size=config.fix_scale_size,
        patch_size=patch_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True,  # 启用锁页内存
        prefetch_factor=2,  # 启用持久化worker
        persistent_workers=True,  # 每个worker预取2个批次
        drop_last=False,  # 是否舍弃最后一个不完整的batch
    )


def load_raod() -> Dataset | DatasetDict:
    raod_dict: Dataset | DatasetDict = load_from_disk(RAOD)
    return raod_dict


if __name__ == "__main__":
    train = load_raod_small(True)
    # sample = train[0]
    # print(sample)
    # build_dataloader()
