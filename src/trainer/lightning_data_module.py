"""用 LightningDataModule 管理 RAOD 数据集与 DataLoader。"""

from typing import Any, override

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import AppConfig
from src.datasets.dataloader import build_dataloader
from src.datasets.raod_benchmark import process_raod_benchmark


class LitHAODDETRDataModule(LightningDataModule):
    """把数据准备和 DataLoader 的生命周期交给 Lightning 管理。

    数据生成/加载在 `raod_benchmark.py`，数据集到 DataLoader 的组装在
    `dataloader.py`；这里只负责在正确的阶段调用它们，并把 loader 交给 Trainer。
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config: AppConfig = config
        self.train_loader: DataLoader[Any] | None = None
        self.val_loader: DataLoader[Any] | None = None

    @override
    def prepare_data(self) -> None:
        # 只在每个节点调用一次，适合做落盘操作；内部已判断缓存是否存在。
        process_raod_benchmark()

    @override
    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self.train_loader is None:
            self.train_loader = build_dataloader(self.config, train=True)
        if self.val_loader is None:
            self.val_loader = build_dataloader(self.config, train=False)

    @override
    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_loader is None:
            self.train_loader = build_dataloader(self.config, train=True)
        return self.train_loader

    @override
    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_loader is None:
            self.val_loader = build_dataloader(self.config, train=False)
        return self.val_loader
