import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from swanlab.integration.pytorch_lightning import SwanLabLogger
from torch.utils.data import DataLoader

from src.config import SwanLabConfig, load_config
from src.datasets.collator import Collator
from src.datasets.raod_benchmark import CLASS_NAMES
from src.trainer.lightning_data_module import LitHAODDETRDataModule
from src.trainer.lightning_module import LitHAODDETR
from src.trainer.prediction import ImagePredictionWriter, PredictionImageDataset
from src.utils.logger import setup_logging


class SwanLabLightningLogger(SwanLabLogger, pl.loggers.Logger):
    """兼容 SwanLab 使用 lightning.pytorch 而训练入口使用 pytorch_lightning。

    两个命名空间的 Logger 类型可能不同；此适配类同时满足当前 Trainer
    的 Logger 类型检查，并复用 SwanLabLogger 的指标及图片记录方法。
    """


class Engine:
    def __init__(
        self, args: dict[str, object] | None = None, *, outputs: str | Path = "outputs"
    ):
        config = load_config(overrides=args)
        # 每次启动都在基础实验名后追加本地时间，避免同名实验难以区分。
        self.config = config
        output_path = Path(outputs).expanduser()
        self.outputs = (
            output_path if output_path.is_absolute() else config.root / output_path
        ).resolve()
        self.outputs.mkdir(parents=True, exist_ok=True)
        setup_logging(self.config.logger)
        self.random_seed(self.config.seed)
        logger.info("配置加载完成：{}", self.config)

    def _checkpoint_path(self, filename: str) -> str:
        return str(self.outputs / "checkpoints" / filename)

    def _experiment(self, config: SwanLabConfig):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config.experiment_name}-{timestamp}"
        return SwanLabLightningLogger(
            project=config.project,
            experiment_name=experiment_name,
            mode="online",
            # 使用绝对路径，确保从其他工作目录启动时输出位置不变。
            log_dir=str(self.outputs / "swanlab"),
            save_dir=str(self.outputs),
        )

    def train(self):
        config = self.config.train_config
        settings = self.config.swanlab
        checkpoint_path = self._checkpoint_path("last.ckpt") if config.resume else None
        if checkpoint_path and not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"恢复训练的检查点不存在：{checkpoint_path}")
        # Lightning 统一设置随机源；workers=True 还会为 DataLoader worker 配置种子。
        # 数据集代码另外给每个 Albumentations 对象设置独立随机种子。
        pl.seed_everything(self.config.seed, workers=True)
        datamodule = LitHAODDETRDataModule(self.config)
        # LightningModule 描述一个批次如何运行；Trainer 在后面驱动完整训练过程。
        task = LitHAODDETR(self.config)
        experiment = self._experiment(settings)
        try:
            # 记录实验配置；后续 self.log 指标通过同一个 logger 写入 SwanLab。
            experiment.log_hyperparams(self.config.model_dump(mode="json"))
            # Callback 是 Trainer 在训练/验证等阶段调用的扩展对象。
            # 每轮验证结束后按 mAP@0.5:0.95 选出 best.ckpt。
            best_checkpoint = ModelCheckpoint(
                dirpath=self.outputs / "checkpoints",
                filename="best",
                monitor="val_map",
                mode="max",
                save_top_k=1,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
            # 单独保存每轮的 last.ckpt，因为 mAP 未刷新时最佳回调可能跳过保存。
            # 放在最佳回调之后，让续训文件包含最新的最佳指标和检查点路径。
            last_checkpoint = ModelCheckpoint(
                dirpath=self.outputs / "checkpoints",
                filename="last",
                save_top_k=1,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
            # max_epochs 为训练总轮数；auto 自动选择可用设备，devices=1 使用单设备。
            # 32-true 使用 float32；LearningRateMonitor 按 step 记录优化器学习率。
            # log_every_n_steps 控制训练步骤日志写入 logger 的频率。
            trainer = pl.Trainer(
                max_epochs=config.epochs,
                accelerator="auto",
                devices=1,
                precision="32-true",
                logger=experiment,
                callbacks=[
                    best_checkpoint,
                    last_checkpoint,
                    LearningRateMonitor(logging_interval="step"),
                ],
                log_every_n_steps=10,
                default_root_dir=self.outputs,
            )
            # fit 负责训练/验证循环、模式切换、自动反向传播、优化器更新和回调。
            # ckpt_path 恢复的不只是模型权重，还包括训练进度及优化器等状态。
            # 配置 resume 为 true 时，从输出目录的 last.ckpt 恢复。
            try:
                trainer.fit(
                    task,
                    datamodule=datamodule,
                    ckpt_path=checkpoint_path,
                )
            except ValueError as exc:
                if checkpoint_path and "parameter group" in str(exc):
                    raise ValueError(
                        "检查点的优化器参数组与当前配置不兼容；旧版单组检查点不能直接断点续训，"
                        "请从头训练或先单独迁移模型权重。"
                    ) from exc
                raise
        finally:
            # 训练结束或发生异常时通知 SwanLab 结束实验。
            experiment.experiment.finish()

    def eval(self):
        checkpoint = self._checkpoint_path("best.ckpt")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"最佳验证检查点不存在：{checkpoint}")
        settings = self.config.swanlab
        pl.seed_everything(self.config.seed, workers=True)
        datamodule = LitHAODDETRDataModule(self.config)
        task = LitHAODDETR(self.config)
        experiment = self._experiment(settings)
        try:
            experiment.log_hyperparams(self.config.model_dump(mode="json"))
            # 独立验证不需要保存新检查点；validate 加载指定权重后只运行验证循环。
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                precision="32-true",
                logger=experiment,
                enable_checkpointing=False,
                default_root_dir=self.outputs,
            )
            # 返回每个验证 DataLoader 的指标字典；verbose=True 同时打印验证结果。
            results = trainer.validate(
                task, datamodule=datamodule, ckpt_path=checkpoint, verbose=True
            )
            logger.info("验证结果：{}", results)
            return results
        finally:
            experiment.experiment.finish()

    def predict(
        self,
        images: Sequence[str | Path],
        *,
        batch_size: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Path]:
        """使用最佳检查点预测本地图片，按输入顺序返回预览路径。"""
        if not images:
            raise ValueError("至少需要一张预测图片")
        options = self.config.predict_config
        if batch_size is None:
            batch_size = options.batch_size
        if score_threshold is None:
            score_threshold = options.score_threshold
        if batch_size < 1:
            raise ValueError("预测批次大小必须为正整数")
        if not 0 <= score_threshold <= 1:
            raise ValueError("预测置信度阈值必须在 [0, 1] 内")

        checkpoint = Path(self._checkpoint_path("best.ckpt"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"最佳预测检查点不存在：{checkpoint}")
        if len(CLASS_NAMES) != self.config.model.num_classes:
            raise ValueError("RAOD 类别名称数量与模型配置的 num_classes 不一致")

        sources = [Path(image).expanduser().absolute() for image in images]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"预测图片不存在：{source}")

        collate = Collator(
            train=False,
            multi_scale=False,
            fix_scale_size=self.config.train_config.fix_scale_size,
            patch_size=self.config.model.patch_size,
            seed=self.config.seed,
        )
        loader = DataLoader(
            PredictionImageDataset(sources),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        task = LitHAODDETR(self.config)
        task.strict_loading = True
        prediction_dir = self.outputs / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(mkdtemp(prefix="run-", dir=prediction_dir))

        writer = ImagePredictionWriter(
            run_dir,
            sources=sources,
            class_names=CLASS_NAMES,
            score_threshold=score_threshold,
        )
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            precision="32-true",
            inference_mode=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[writer],
            default_root_dir=self.outputs,
        )
        trainer.predict(
            task,
            dataloaders=loader,
            ckpt_path=checkpoint,
            weights_only=True,
            return_predictions=False,
        )
        return writer.saved_paths

    def random_seed(self, seed: int = 42) -> None:
        """固定全局随机源；在创建模型、DataLoader 和执行随机操作之前调用。

        NumPy 的旧式全局随机源要求种子处于 [0, 2**32) 范围。
        Albumentations Compose、np.random.default_rng 等独立生成器仍需单独设种子。
        相同种子有助于复现实验，不保证跨设备、跨版本的计算结果完全一致。
        """
        if not 0 <= seed < 2**32:
            raise ValueError("seed 必须处于 [0, 2**32) 范围")

        random.seed(seed)  # Python 随机操作，例如 random.choice。
        np.random.seed(seed)  # NumPy 全局随机操作。
        torch.manual_seed(seed)  # PyTorch CPU 及加速设备（包括 CUDA、MPS）。

        # CUDA 卷积关闭算法测速，优先使用确定性算法；这不约束其他算子。
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
