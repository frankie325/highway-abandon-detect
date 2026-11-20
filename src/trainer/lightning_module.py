"""Lightning 训练封装：模型、匹配损失和验证可视化。"""

import math

import pytorch_lightning as pl
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import AppConfig
from src.datasets.types import DetectionBatch
from src.loss.criterion import SetCriterion
from src.loss.matcher import build_matcher
from src.models import build_model
from src.trainer.metrics import (
    DetectionConfusionMatrix,
    draw_confusion_matrix,
    draw_pr_curve,
    prepare_detection_inputs,
)
from src.trainer.param_groups import build_param_dicts
from src.trainer.visualization import detection_previews


class LitHAODDETR(pl.LightningModule):
    """把模型和检测任务交给 Lightning 的训练循环管理。

    LightningModule 也是 torch.nn.Module。这里定义单批次如何计算损失、
    如何验证以及使用什么优化器；Trainer 负责循环、反向传播和参数更新。
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        # 注册到 LightningModule 的子模块会随整个模块切换 train/eval 和设备。
        # criterion 也需要切换状态，以便 Group DETR 在训练和验证时使用正确的分组数。
        self.model = build_model(config)
        self.criterion = SetCriterion(
            num_classes=config.model.num_classes,
            matcher=build_matcher(config),
            focal_alpha=config.train_config.focal_alpha,
            group_detr=config.model.group_detr,
            ia_bce_loss=True,
        )
        self.loss_weights = config.train_config.loss_weights.model_dump()
        # COCO 风格 mAP 需要累积整个验证集；扩展结果中的 precision 用于绘制 PR 曲线。
        self.val_map = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", extended_summary=True
        )
        self.val_confusion = DetectionConfusionMatrix(
            num_classes=config.model.num_classes,
            score_threshold=config.evaluate_config.confusion_score_threshold,
            iou_threshold=config.evaluate_config.confusion_iou_threshold,
        )

    def forward(self, samples):
        """定义 task(samples) 的前向推理；训练步骤也可以直接调用 self.model。"""
        return self.model(samples)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Trainer 在 step 之前调用，将自定义批次移动到当前设备。

        原图始终留在 CPU。训练和验证需要把标注移到模型设备，
        预测时会在 CPU 上解码检测框，因此空标注也留在 CPU。
        """
        samples, targets, *originals = batch
        if self._trainer is None or not self._trainer.predicting:
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
        return samples.to(device), targets, *originals

    def predict_step(
        self, batch: DetectionBatch, batch_idx: int
    ) -> list[dict[str, torch.Tensor]]:
        """对一批本地图像做前向推理，解码后交给逐批保存回调。"""
        samples, empty_targets, _ = batch
        predictions, _ = prepare_detection_inputs(
            samples,
            empty_targets,
            self(samples),
            max_detections=self.config.predict_config.max_detections,
        )
        return predictions

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """主输出、辅助层及 encoder 损失使用各自基础名称对应的权重。"""
        weighted = []
        for name, value in losses.items():
            matches = [
                weight
                for base, weight in self.loss_weights.items()
                if name == base or name.startswith(base + "_")
            ]
            if len(matches) != 1:
                raise ValueError(f"未配置损失权重: {name}")
            weighted.append(value * matches[0])
        return torch.stack(weighted).sum()

    def _step(self, batch, stage: str):
        """训练与验证共用的单批次计算；是否求梯度由 Trainer 管理。"""
        samples, targets = batch[:2]
        outputs = self.model(samples)
        losses = self.criterion(outputs, targets)
        total = self.weighted_loss(losses)
        # self.log 将指标交给 Trainer 和配置的 logger（这里是 SwanLab）。
        # 训练同时记录每步和每轮；验证只记录每轮。val_loss 供检查点回调监控。
        # prog_bar=True 显示在进度条中；batch_size 用于按批次大小聚合 epoch 指标。
        # sync_dist=True 在分布式运行时跨进程同步；当前项目使用单设备。
        self.log(
            f"{stage}_loss",
            total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )
        # log_dict 一次记录多项指标；这里只单独展示最终层的三项基础损失。
        # 辅助层和 encoder 损失仍包含在上面的 total 中。
        self.log_dict(
            {f"{stage}_{name}": losses[name] for name in self.loss_weights},
            on_step=stage == "train",
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )
        return total, outputs

    def training_step(self, batch, batch_idx):
        """Trainer 对每个训练批次调用；返回标量损失用于自动反向传播。

        在默认自动优化模式下，无需手动调用 backward()、optimizer.step()
        或 optimizer.zero_grad()。batch_idx 是当前 epoch 内的批次序号。
        """
        total, _ = self._step(batch, "train")
        return total

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """记录当前优化器的全局梯度 L2 范数。"""
        # 此时梯度已累积并还原 AMP 缩放，Lightning 随后才执行裁剪。
        gradients = [
            param.grad
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if gradients:
            total_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        """Trainer 在验证阶段逐批调用；Lightning 负责切换评估模式并禁用梯度。"""
        _, outputs = self._step(batch, "val")
        options = self.config.evaluate_config
        if not self.trainer.sanity_checking:
            # 所有验证批次都参与指标计算；预检不加入正式验证结果。
            # 验证时 outputs["pred_logits"]：[B, Q, C]，outputs["pred_boxes"]：[B, Q, 4]；
            # Q 是验证阶段使用的 Query 数，C 是前景类别数。targets 是长度为 B 的变长列表。
            predictions, targets = prepare_detection_inputs(
                batch[0],
                batch[1],
                outputs,
                max_detections=options.max_detections,
            )
            self.val_map.update(predictions, targets)
            self.val_confusion.update(predictions, targets)
        # 每轮仅记录第一个批次；多进程时只由主进程记录，避免重复上传。
        # sanity_checking 表示 fit 开始前的验证预检，跳过它以免产生额外图片。
        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and not self.trainer.sanity_checking
        ):
            images = detection_previews(
                batch[0],
                batch[1],
                outputs,
                score_threshold=options.preview_score_threshold,
                max_predictions=options.preview_max_predictions,
            )
            # log_image 是当前 SwanLab logger 的接口；global_step 是优化器更新步数。
            self.logger.log_image(
                key="val/predictions", images=images, step=self.global_step
            )

    def on_validation_epoch_end(self):
        """整轮验证后统一计算 mAP 和绘制图表，避免逐 batch 求平均。"""
        if self.trainer.sanity_checking:
            self.val_map.reset()
            self.val_confusion.reset()
            return
        if not self.val_map.update_called:
            return
        try:
            # 所有进程都参与 compute；TorchMetrics 会先同步状态。
            results = self.val_map.compute()
            matrix = self.val_confusion.compute()
            self.log_dict(
                {
                    "val_map": results["map"],
                    "val_map_50": results["map_50"],
                    "val_map_75": results["map_75"],
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # is_global_zero 表示当前进程的全局编号（global_rank）为 0；单进程训练时为 True。
            # 所有进程都参与上面的指标计算，但只由 0 号进程绘制并记录图表，避免 SwanLab 重复记录。
            if self.trainer.is_global_zero:
                self.logger.log_image(
                    key="val/pr_curve",
                    images=[draw_pr_curve(results)],
                    step=self.global_step,
                )
                self.logger.log_image(
                    key="val/confusion_matrix",
                    images=[
                        draw_confusion_matrix(
                            matrix,
                            score_threshold=self.config.evaluate_config.confusion_score_threshold,
                            iou_threshold=self.config.evaluate_config.confusion_iou_threshold,
                        )
                    ],
                    step=self.global_step,
                )
        finally:
            self.val_map.reset()
            self.val_confusion.reset()

    def _build_optimizer(self):
        """根据配置构建优化器，调度器将在 configure_optimizers 中接入。"""
        train_config = self.config.train_config
        # 按检测头、Decoder 和编码器深度分组；冻结的参数不交给优化器。
        parameters = build_param_dicts(
            self.config,
            self.model,
        )
        options = {
            "lr": train_config.lr,
            "weight_decay": train_config.weight_decay,
        }
        if train_config.optimizer == "sgd":
            # 动量累积历史梯度，减少 SGD 更新过程中的震荡。
            return torch.optim.SGD(
                parameters, momentum=train_config.momentum, **options
            )
        if train_config.optimizer == "adam":
            return torch.optim.Adam(parameters, **options)
        if train_config.optimizer == "adamw":
            return torch.optim.AdamW(parameters, **options)
        raise ValueError(f"不支持的优化器: {train_config.optimizer}")

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, object]:
        """Step 按 epoch 更新；Cosine 按优化器步数执行 Warmup 和余弦衰减。"""
        config = self.config.train_config
        if config.lr_scheduler == "step":
            # 每经过 step_size 个 epoch，将每个参数组的学习率乘以 step_gamma。
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.step_gamma
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        if config.lr_scheduler != "cosine":
            raise ValueError(f"不支持的学习率调度策略: {config.lr_scheduler}")
        # Lightning 的估算考虑训练批次、梯度累积和 max_steps 等 Trainer 设置。
        total_steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(total_steps) or total_steps < 1:
            raise ValueError("学习率调度需要至少一个有限的优化器训练步数")
        total_steps = int(total_steps)
        # 至少保留一个余弦衰减步；只有一步时直接使用基础学习率训练。
        warmup_steps = min(
            math.ceil(total_steps * config.warmup_ratio), total_steps - 1
        )

        # 学习率倍率，group[’lr’] = base_lr × lr_factor(step)
        # base_lr为每个param_group设置的lr
        def lr_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                # 初始学习率为基础学习率 / 预热步数，不让首次更新的学习率为 0。
                return (step + 1) / warmup_steps
            progress = min(
                max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        return {"scheduler": scheduler, "interval": "step"}

    def configure_optimizers(self):
        """为 Lightning 返回优化器和调度器，按策略配置更新间隔。"""
        optimizer = self._build_optimizer()
        scheduler = self.build_lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
