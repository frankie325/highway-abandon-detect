import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from functools import partial
from typing import Any

from .collator import Collator
from .raod_benchmark import load_raod_small, load_raod
from src.config import AppConfig


def seed_raod_worker(worker_id: int, *, collator: Collator) -> None:
    """让每个 worker 的 Albumentations 独立随机且可复现。"""
    seed = torch.initial_seed() % (2**32)
    for index, transform in enumerate(collator.transforms.values()):
        transform.set_random_seed((seed + index * 0x9E3779B9) % (2**32))


def build_dataloader(config: AppConfig, train: bool) -> DataLoader[Any]:
    train_config = config.train_config
    model_config = config.model

    collator = Collator(
        train=train,
        multi_scale=train_config.multi_scale,
        scale_sizes=train_config.scale_sizes,
        fix_scale_size=train_config.fix_scale_size,
        patch_size=model_config.patch_size,
        seed=config.seed,
    )

    load_dataset = load_raod if train_config.dataset == "raod" else load_raod_small
    dataset: Dataset = load_dataset(train)

    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=train,
        num_workers=train_config.num_workers,
        collate_fn=collator,
        pin_memory=True,  # 启用锁页内存
        prefetch_factor=2 if train_config.num_workers > 0 else None,
        persistent_workers=train_config.num_workers > 0,
        worker_init_fn=partial(seed_raod_worker, collator=collator),
        generator=torch.Generator().manual_seed(config.seed + (0 if train else 1)),
        drop_last=False,  # 是否舍弃最后一个不完整的batch
    )
