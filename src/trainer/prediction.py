"""本地预测图片读取和 Lightning 逐批结果保存。"""

from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from PIL import Image, ImageOps
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import Dataset

from src.datasets.types import DataSample, DetectionBatch
from src.trainer.visualization import draw_prediction_boxes


class PredictionImageDataset(Dataset[DataSample]):
    """按输入顺序读取本地图片，提供 Collator 接收的单图样本。"""

    def __init__(self, sources: Sequence[Path]) -> None:
        self.sources = sources

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> DataSample:
        source = self.sources[index]
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise ValueError(f"无法解码预测图片：{source} ({exc})") from exc
        return {
            "image": image,
            "image_id": source.stem,
            "width": image.width,
            "height": image.height,
            "boxes": [],
            "labels": [],
        }


class ImagePredictionWriter(BasePredictionWriter):
    """每批推理结束后将检测结果绘制到原图并保存。"""

    def __init__(
        self,
        run_dir: Path,
        *,
        sources: Sequence[Path],
        class_names: Sequence[str],
        score_threshold: float,
    ) -> None:
        super().__init__(write_interval="batch")
        self.run_dir = run_dir
        self.sources = sources
        self.class_names = class_names
        self.score_threshold = score_threshold
        self.saved_paths: list[Path] = []

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: list[dict[str, torch.Tensor]],
        batch_indices: Sequence[int] | None,
        batch: DetectionBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        samples, _, originals = batch
        if batch_indices is None:
            raise RuntimeError("Lightning 未提供预测图片的批次索引")
        mask = samples.mask
        assert mask is not None
        valid_widths = (~mask[:, 0, :]).sum(dim=1)
        valid_heights = (~mask[:, :, 0]).sum(dim=1)
        resized_sizes = torch.stack((valid_widths, valid_heights), dim=1).tolist()

        for index, original, detected, size in zip(
            batch_indices, originals, prediction, resized_sizes, strict=True
        ):
            source = self.sources[index]
            image = original["image"]
            count = draw_prediction_boxes(
                image,
                detected,
                (size[0], size[1]),
                class_names=self.class_names,
                score_threshold=self.score_threshold,
            )
            destination = self.run_dir / f"{index + 1:04d}_{source.stem}.png"
            image.save(destination)
            self.saved_paths.append(destination)
            logger.info("{}：检测到 {} 个目标，已保存到 {}", source, count, destination)
