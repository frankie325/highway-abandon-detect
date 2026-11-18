from pathlib import Path
from functools import partial
from datasets import load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader
from loguru import logger
import torch

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


def seed_raod_worker(worker_id: int, *, collator: RAODCollate) -> None:
    """让每个 worker 的 Albumentations 独立随机且可复现。"""
    seed = torch.initial_seed() % (2**32)
    for index, transform in enumerate(collator.transforms.values()):
        transform.set_random_seed((seed + index * 0x9E3779B9) % (2**32))


def build_dataloader(
    train: bool, config: TrainConfig, *, patch_size: int, seed: int = 42
) -> DataLoader:
    dataset: Dataset = load_raod_small(train)
    collator = RAODCollate(
        train=train,
        multi_scale=config.multi_scale,
        scale_sizes=config.scale_sizes,
        fix_scale_size=config.fix_scale_size,
        patch_size=patch_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True,  # 启用锁页内存
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=partial(seed_raod_worker, collator=collator),
        generator=torch.Generator().manual_seed(seed + (0 if train else 1)),
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
