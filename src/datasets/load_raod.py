from pathlib import Path

from datasets import load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader
from src.config import TrainConfig
from src.datasets.collate import RAODCollate
from loguru import logger

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


def load_raod_small(train: bool = True) -> Dataset:
    logger.info("加载raod_small数据集")
    ds = load_from_disk(RAOD_SMALL)
    small_dict: DatasetDict = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    return small_dict["train" if train else "test"]


def build_dataloader(train: bool, args: TrainConfig, *, patch_size: int) -> DataLoader:
    dataset: Dataset = load_raod_small(train)
    collator = RAODCollate(
        train=train,
        multi_scale=args.multi_scale,
        scale_sizes=args.scale_sizes,
        fix_scale_size=args.fix_scale_size,
        patch_size=patch_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )


def load_raod() -> Dataset | DatasetDict:
    raod_dict: Dataset | DatasetDict = load_from_disk(RAOD)
    return raod_dict


if __name__ == "__main__":
    train = load_raod_small(True)
    # Hugging Face Dataset 不携带单条样本的泛型类型，取出时显式标注。
    # sample = train[0]
    # print(sample)
    # build_dataloader()
