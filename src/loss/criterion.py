from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any
from typing import Any, NamedTuple, Union

from .matcher import HungarianMatcher
from src.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
)

_LossFunction = Callable[..., dict[str, torch.Tensor]]


class _MatchedTargets(NamedTuple):
    """Indices and target tensors shared by detection losses for one output layer.

    ``loss_labels`` and ``loss_boxes`` consume the same matched labels, boxes, and source indices. Keeping them together
    avoids rebuilding those tensors for each loss without changing their per-layer lifetime or ordering.
    """

    source_indices: tuple[torch.Tensor, torch.Tensor]
    labels: torch.Tensor
    boxes: torch.Tensor


# ====================== 分类损失：公共约定 ======================
# 此文件调用时 inputs 与 targets 的形状均为 [B, Q, C]：
# B 为 batch 大小，Q 为 query 数，C 为前景类别数；各类别独立使用 sigmoid。
# z 是 inputs 中的 logit，p 是预测概率，t 是当前位置的目标：
# $$ p=\sigma(z) $$
# 逐元素二元交叉熵（BCE）支持 t 为 0、1 或 [0,1] 中的软目标：
# $$ \operatorname{BCE}(p,t)=-t\log p-(1-t)\log(1-p) $$
# 三个函数均返回 loss.mean(1).sum() / num_boxes。对 [B,Q,C]：
# mean(1) 对 Q 求平均，sum() 对剩余 B、C 求和；loss_labels() 的调用
# 再乘 Q，因此该调用路径最终等价于对 B、Q、C 全部求和，再除以 num_boxes。
# num_boxes 由调用方计算；group_detr 和分布式归一化见 num_boxes_for_targets()。
# ==================================================================


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    r"""Sigmoid Focal Loss：降低分类容易样本的权重。

    要解决的问题：目标检测中负样本通常远多于正样本。即使每个简单背景
    的 BCE 很小，大量背景加起来仍可能主导训练，让模型更关注背景而非目标。
    Focal Loss 根据当前预测降低容易样本的权重，使困难样本有更大的相对影响。

    输入：inputs 为 logits，targets 为同形状标签；本项目调用时是 [B,Q,C]，
    匹配 Query 的真实类别为 1，其他 Query × 类别位置为 0。
    令 $$ p=\sigma(z) $$，对于单个位置，先定义：
    $$ p_t=pt+(1-p)(1-t) $$
    $$ \alpha_t=\alpha t+(1-\alpha)(1-t) $$
    $$ L_{\mathrm{focal}}=\alpha_t(1-p_t)^\gamma\operatorname{BCE}(p,t) $$

    正样本 t=1 和负样本 t=0 时，展开分别为：
    $$ L_{\mathrm{pos}}=-\alpha(1-p)^\gamma\log p $$
    $$ L_{\mathrm{neg}}=-(1-\alpha)p^\gamma\log(1-p) $$
    正样本的 p 越接近 1、负样本的 p 越接近 0，其权重越小。
    例如 alpha=0.25、gamma=2 时，p=0.9 的正样本调制系数为 0.01。
    alpha < 0 时代码跳过 alpha_t；gamma=0 时只剩 BCE 和可选的 alpha 平衡。

    对应代码：p_t 对应 p_t，(1-p_t)**gamma 是调制因子，alpha_t 是类别权重。
    返回值的维度归约见上方公共约定；本项目的调用处额外乘 Q。
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    result: torch.Tensor = loss.mean(1).sum() / num_boxes
    return result


def sigmoid_varifocal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    r"""Sigmoid Varifocal Loss：用框的 IoU 监督分类分数。

    要解决的问题：分类分数高的预测框不一定定位准确，仅按分类分数排序时，
    较差的框可能排在定位更好的框前面。Varifocal Loss 将定位质量引入
    分类目标，希望分类分数有助于对候选框质量排序，同时降低容易负样本的权重。

    输入和形状见上方公共约定。本项目的调用处把匹配框 IoU 填入真实类别位置，
    其他位置为 0；令 t=q、q 为 IoU，p=sigmoid(inputs)：

    $$ w=\begin{cases}q,&q>0\\(1-\alpha)p^\gamma,&q=0\end{cases} $$


    $$ L_{\mathrm{VFL}}=w\operatorname{BCE}(p,q) $$

    展开可得：
    $$ L_{q>0}=-q\left[q\log p+(1-q)\log(1-p)\right] $$
    $$ L_{q=0}=-(1-\alpha)p^\gamma\log(1-p) $$
    例如 q=0.6 时，BCE 的目标值和正分支权重均为 0.6；负分支则按预测
    概率 p 的 gamma 次方调制。本实现的负项系数是 1-alpha。

    注意分支条件是 targets > 0：匹配框 IoU 恰为 0 时，数值目标仍为 0，
    本函数按负分支计算。调用处构造 IoU 目标时已 detach，不通过该目标反传到框。
    focal_weight 对应 w；返回值归约见公共约定，调用处额外乘 Q。
    """
    prob = inputs.sigmoid()
    focal_weight = (
        targets * (targets > 0.0).float()
        + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float()
    )
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    r"""Position-Supervised Loss：根据分类概率与位置质量目标的偏差调制 BCE。

    要解决的问题：如果分类只学习目标类别，分类分数可能无法体现框的定位质量；
    使用 IoU 等软目标时，还需要衡量预测分数与质量目标之间的差距。
    此实现用该差距调制损失，强调预测分数偏离位置质量目标较多的位置。

    输入和形状见上方公共约定。调用处先将匹配框 IoU 写入真实类别位置，
    再按每张图片的所有 Query × 类别位置的最大 IoU 归一化：

    $$ t_{bqc}=\frac{\operatorname{IoU}_{bqc}}{\max_{q',c'}\operatorname{IoU}_{bq'c'}+10^{-8}} $$


    未匹配的位置先设为 0；IoU 目标在调用处 detach。
    对单个位置，令 p=sigmoid(inputs)，代码计算：


    $$ \alpha_t=\begin{cases}\alpha,&t>0\\1-\alpha,&t\leq 0\end{cases} $$


    $$ L_{\mathrm{PSL}}=\alpha_t|t-p|^\gamma\operatorname{BCE}(p,t) $$

    例如 t=0.6、p=0.8、gamma=2 时，调制因子是 0.04；它衡量的是
    预测概率与归一化位置质量目标的差距。t=0 时使用负样本权重；匹配框
    的 IoU 若为 0，其对应位置也会数值上走负分支。
    当 t 只取 0 或 1 时，本实现与 sigmoid_focal_loss 数值等价；alpha < 0
    时跳过 alpha_t。返回值归约见公共约定，调用处额外乘 Q。
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = (
            alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        )
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        # weight_dict: dict[str, float],
        focal_alpha: float,
        group_detr: int = 1,
        sum_group_losses: bool = False,  # 损失是否要对group求平均
        use_varifocal_loss: bool = False,  # 是否使用varifocal loss
        use_position_supervised_loss: bool = False,  # 是否使用position supervised loss
        ia_bce_loss: bool = False,  # 是否使用ia bce loss
        # mask_point_sample_ratio: int = 16,
        # num_keypoints_per_class: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # self.weight_dict = weight_dict
        self.losses = ["labels", "boxes"]
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        # self.mask_point_sample_ratio = mask_point_sample_ratio
        # self.num_keypoints_per_class = num_keypoints_per_class or []

    @staticmethod
    def _output_device(outputs: dict[str, Any]) -> torch.device:
        """Return the device used by tensor outputs.

        Args:
            outputs: Model output dictionary. Top-level values are probed for tensors;
                nested structures (lists, nested dicts) are not traversed.

        Returns:
            Device of the first tensor value found in ``outputs``.

        Raises:
            ValueError: If no tensor output is present.
        """
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.device
        raise ValueError(
            "SetCriterion requires at least one tensor output to infer the loss device."
        )

    def num_boxes_for_targets(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute the distributed target-box denominator for a target batch.

        The denominator is the total number of ground-truth boxes in the batch, multiplied by the active number of
        DETR groups (unless ``sum_group_losses`` collapses them), reduced across all distributed ranks, divided by the
        world size, and finally clamped to be at least ``1.0`` so divide-by-zero never occurs on empty batches.

        Args:
            outputs: Model output dictionary; used only to infer the device for the
                returned scalar tensor.
            targets: Per-image target dictionaries for the current batch. Each must
                contain a ``"labels"`` tensor whose length equals the number of
                ground-truth boxes for that image.

        Returns:
            Scalar tensor on the same device as the model outputs, holding the
            average box-count denominator used to normalize criterion losses.

        Note:
            When ``torch.distributed`` is initialized this method performs an
            in-place ``all_reduce`` collective on the returned tensor. Every rank
            must reach this call together or the program will deadlock.

        Note:
            ``group_detr`` is multiplied in only when ``self.training`` is ``True``.
            During evaluation (``self.training`` is ``False``) the denominator
            collapses to a single group, so train-time and eval-time normalizers
            cannot be compared directly.

        Examples:
            >>> import torch
            >>> from rfdetr.models.criterion import SetCriterion
            >>> criterion = SetCriterion.__new__(SetCriterion)
            >>> criterion.training = False
            >>> criterion.group_detr = 1
            >>> criterion.sum_group_losses = False
            >>> outputs = {"pred_logits": torch.zeros(1, 1, 1)}
            >>> targets = [{"labels": torch.tensor([0, 1, 2])}]
            >>> criterion.num_boxes_for_targets(outputs, targets).item()
            3.0
        """
        group_detr = self.group_detr if self.training else 1
        # 当前批次所有真实框的数量
        num_boxes = sum(len(t["labels"]) for t in targets)

        # 分组损失是否要平均
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes_tensor = torch.as_tensor(
            num_boxes, dtype=torch.float, device=self._output_device(outputs)
        )
        return num_boxes_tensor

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        展平prediction_indices和target_indices
        输出为(batch_idx, src_idx)
        batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
        src_idx = [0,1,0,1,0,1,2,...] [样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
        """
        # permute predictions following indices
        # 输出为[0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        # 输出[样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_matched_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> _MatchedTargets:
        """Collect matched detection targets for losses sharing one layer's indices.

        Args:
            targets: Per-image target dictionaries in batch order.
            indices: Per-image matcher results, where each pair contains source
                query indices and their corresponding target indices.

        Returns:
            Matched targets containing ``source_indices``, ``labels``, and
            ``boxes``. The tensors are concatenated across images in the same
            order as ``indices`` and ``targets``, so all three fields use the
            same flattened batch-of-matches ordering. Callers must not reorder
            one field without applying the same reordering to the others.
        """

        """
        targets = list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        indices = [
                    # 第一个样本
                    (
                        prediction_indices, 每个组的prediction_indices 拼接在一起
                        target_indices,  每个组的target_indices 拼接
                    ),
                    ...
                ]

        得到
        source_indices = (batch_idx, src_idx)
            batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
            src_idx = [0,1,0,1,0,1,2,...] [样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
        labels = [ 0, 2, 2, 4, ...] 所有样本的匹配框类别值
        boxes = [(cx, cy, w, h), ...] 所有样本的匹配框坐标值
        """
        return _MatchedTargets(
            source_indices=self._get_src_permutation_idx(indices),
            labels=torch.cat(
                [
                    target["labels"][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ]
            ),
            boxes=torch.cat(
                [
                    target["boxes"][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ]
            ),
        )

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        group_detr = self.group_detr if self.training else 1

        """
        outputs = {
            # 最后一层输出
            "pred_logits": output_classes[-1],  # b, num_queries * group_detr, num_classes
            "pred_boxes": output_boxes[-1],  # b, num_queries * group_detr, 4
            # 中间辅助层输出
            "aux_outputs": [
                {
                    "pred_logits": output_classes[i],  # b, num_queries * group_detr, num_classes
                    "pred_boxes": output_boxes[i],  # b, num_queries * group_detr, 4
                }
            ]
            # 两阶段时编码器输出
            "enc_outputs": {
                "pred_logits": cls_enc,  # b, topk * group_detr, num_classes
                "pred_boxes": ref_enc,  # b, topk * group_detr, 4
            }
        }
        targets = list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        matched_outputs = [outputs_without_aux]
        layer_suffixes = [""]

        # matched_outputs 中每个对象为 [{ "pred_logits": pred_logits, "pred_boxes": pred_boxes },...]
        if "aux_outputs" in outputs:
            matched_outputs.extend(outputs["aux_outputs"])
            layer_suffixes.extend(
                f"_{aux_index}" for aux_index in range(len(outputs["aux_outputs"]))
            )

        if "enc_outputs" in outputs:
            matched_outputs.append(outputs["enc_outputs"])
            layer_suffixes.append("_enc")

        """
        计算所有层的匹配结果
        [
            # 最后一个层的匹配结果
            [
                # 第一个样本
                (
                    prediction_indices, 每个组的prediction_indices 拼接在一起
                    target_indices,  每个组的target_indices 拼接
                ),
                ...
            ]
            # 辅助层
            [
                # 第一个样本
                (
                    prediction_indices, 每个组的prediction_indices 拼接在一起
                    target_indices,  每个组的target_indices 拼接
                ),
                ...
            ]
            ...
            # 两阶段时编码器输出层
            [
                # 第一个样本
                (
                    prediction_indices, 每个组的prediction_indices 拼接在一起
                    target_indices,  每个组的target_indices 拼接
                ),
                ...
            ]
        ]
        """
        all_indices = [
            self.matcher(layer_outputs, targets) for layer_outputs in matched_outputs
        ]

        losses = {}

        num_boxes = self.num_boxes_for_targets(outputs, targets)
        for suffix, layer_outputs, indices in zip(
            layer_suffixes, matched_outputs, all_indices, strict=True
        ):
            matched_targets = self._get_matched_targets(targets, indices)
            """
            losses = ["labels", "boxes"]
            layer_outputs = { "pred_logits": pred_logits, "pred_boxes": pred_boxes }
            targets = list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
            indices = [
                # 第一个样本
                (
                    prediction_indices, 每个组的prediction_indices 拼接在一起
                    target_indices,  每个组的target_indices 拼接
                ),
                ...
            ]
            matched_targets = {
                source_indices = (batch_idx, src_idx)
                    - batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
                    - src_idx = [0,1,0,1,0,1,2,...] [样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
                labels = [ 0, 2, 2, 4, ...] 所有样本的匹配框类别值
                boxes = [(cx, cy, w, h), ...] 所有样本的匹配框坐标值
            }
            """
            for loss in self.losses:
                layer_losses = self.get_loss(
                    loss, layer_outputs, targets, indices, num_boxes, matched_targets
                )
                losses.update(
                    {key + suffix: value for key, value in layer_losses.items()}
                )

        # 汇总打印全部损失（各 decoder 层带后缀的 loss + 合计），便于调试
        # 每个 loss 的量级与占比。losses 形如
        # {"loss_ce_0": tensor(...), "loss_ce_1": tensor(...), ...}
        # 用 lazy=True：仅当确实有 sink 会处理该 DEBUG 记录时才求值，
        # 避免日志整体关闭 DEBUG 时仍执行 .item() 触发 GPU→CPU 同步。
        # 注意 loguru 的 lazy 要求所有格式化参数都必须是 callable。
        # 项目默认 file_level="DEBUG"（见 src/config.py）：日志文件每步都会
        # 记录这条汇总，而终端 level="INFO" 默认不显示，需调到 DEBUG 才可见。
        logger.opt(lazy=True).debug(
            "全部损失 ({count} 项, total={total}): {values}",
            count=lambda: len(losses),
            total=lambda: round(
                sum(value.detach().item() for value in losses.values()), 6
            ),
            values=lambda: {
                name: round(value.detach().item(), 6) for name, value in losses.items()
            },
        )
        return losses

    def loss_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        matched_targets: _MatchedTargets,
        # log: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        losses = ["labels", "boxes"]
        layer_outputs = { "pred_logits": pred_logits, "pred_boxes": pred_boxes }
        targets = list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        indices = [
            # 第一个样本
            (
                prediction_indices, 每个组的prediction_indices 拼接在一起
                target_indices,  每个组的target_indices 拼接
            ),
            ...
        ]
        matched_targets = {
            source_indices = (batch_idx, src_idx)
                - batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
                - src_idx = [0,1,0,1,0,1,2,...] [样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
            labels = [ 0, 2, 2, 4, ...] 所有样本的匹配框类别值
            boxes = [(cx, cy, w, h), ...] 所有样本的匹配框坐标值
        }
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # b, num_queries * group_detr, num_classes

        idx = matched_targets.source_indices
        target_classes_o = matched_targets.labels
        target_boxes = matched_targets.boxes

        if self.ia_bce_loss:
            # IA-BCE（IoU-aware BCE）是分类损失：让分类分数同时反映类别置信度和框的定位质量。
            # 要解决的问题：分类得分与框的定位质量可能不一致，导致按分数排序时
            # 高分框不一定定位更准；同时大量容易负样本可能占据分类损失。
            # 这里用概率和 IoU 共同构造软目标，并降低容易负样本的权重。
            #
            # 1. 确定正负样本
            # 判断单位是「图片 × Query × 类别」：
            # - 匹配成功的 Query，其真实类别位置是正样本，其他类别位置仍是负样本。
            # - 未匹配 Query 的所有类别位置都是负样本；正负由匹配结果决定，而非固定 IoU 阈值。
            #
            # 2. 为匹配到的真实类别构造软目标
            # 对应分类 logit z、概率 p 和匹配框 IoU u：
            # $$ p=\sigma(z),\quad u=\operatorname{IoU}(\hat{b},b),\quad t=\max\left(p^\alpha u^{1-\alpha},\,0.01\right) $$
            # 例如 alpha=0.25、p=0.8、u=0.2 时，t≈0.283。代码对 t 调用 detach()，
            # 因此本次反向传播将 t 视为常量；下一次前向传播仍会重新计算。
            #
            # 3. 对所有「Query × 类别」位置计算分类损失
            # $$ L_{\mathrm{pos}}=-t\log p-(1-t)\log(1-p) $$
            # $$ L_{\mathrm{neg}}=-p^\gamma\log(1-p) $$
            # 负样本的 $$ p^\gamma $$ 降低容易负样本的权重；这里 gamma=2，且该权重保留梯度。
            # 分类损失不会通过软目标中的 IoU 更新预测框；框由 L1 和 GIoU 损失监督。
            alpha = self.focal_alpha
            gamma = 2

            """
            pred_boxes: b, num_queries * group_detr, 4
            idx = (batch_idx, src_idx) 
                batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
                  src_idx = [0,1,0,1,0,1,2,...]
            相当于
            pred_boxes[0, 0, :]
            pred_boxes[0, 1, :]
            通过数组索引拿到对应的预测框

            src_boxes: M, 4
              M = batch 内所有匹配对总数 = group_detr * sum(每个样本匹配框数)
            """
            src_boxes = outputs["pred_boxes"][idx]
            # M,
            iou_targets, _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes),
            )

            # M,
            pos_ious = iou_targets.clone().detach()

            # 得到概率
            prob = src_logits.sigmoid()

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)  # [batch_idx, src_idx, target_classes_o]
            # 使用高级数组索引，取到预测框对应的类别概率  M,
            p = prob[tuple(pos_ind)]

            # M,
            t = p.pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights = torch.zeros_like(
                src_logits
            )  # b, num_queries * group_detr, num_classes
            # !先把所有 Query × 类别位置设为负样本
            neg_weights = prob**gamma

            # !仅把“匹配成功的 Query × 真实类别”改成正样本
            pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
            neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)

            # 4. 将正负位置合并，并用 logits 计算更稳定的形式。
            # 令 a=pos_weights、b=neg_weights，分别为 BCE 正项和负项的权重：
            # 正样本 $$ (a,b)=(t,1-t) $$；负样本 $$ (a,b)=(0,p^\gamma) $$。
            # 首先由 sigmoid 的定义得到概率与 logit 的关系：
            # $$ p=\sigma(z)=\frac{1}{1+e^{-z}} $$
            # $$ \frac{p}{1-p}=e^z $$
            # 两边取对数并移项：
            # $$ \log p-\log(1-p)=z $$
            # $$ \log(1-p)=\log p-z $$
            # 将此关系代入逐位置的加权 BCE：
            # $$ L=-a\log p-b\log(1-p) $$
            # $$ L=-a\log p-b(\log p-z) $$
            # $$ L=bz-(a+b)\log p $$
            # 由于 $$ \log p=\operatorname{logsigmoid}(z) $$，得到代码中的稳定写法：
            # $$ L=bz-(a+b)\operatorname{logsigmoid}(z) $$
            # 最后对 batch、Query 和类别求和，并除以 num_boxes。
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (
                pos_weights + neg_weights
            )
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs["pred_boxes"][idx]
            iou_targets, _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes),
            )
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            # b, num_queries * group_detr, num_classes
            cls_iou_func_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)  # [batch_idx, src_idx, target_classes_o]
            pos_ious_func = pos_ious_func.to(cls_iou_func_targets.dtype)

            # 匹配位置设置iou
            cls_iou_func_targets[tuple(pos_ind)] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets / (
                cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(
                    1, True
                )
                + 1e-8
            )
            loss_ce = (
                position_supervised_loss(
                    src_logits,
                    norm_cls_iou_func_targets,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )

        elif self.use_varifocal_loss:
            src_boxes = outputs["pred_boxes"][idx]

            iou_targets, _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes),
            )
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            cls_iou_targets[tuple(pos_ind)] = pos_ious
            loss_ce = (
                sigmoid_varifocal_loss(
                    src_logits,
                    cls_iou_targets,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )
        else:
            # src_logits: b, num_queries * group_detr, num_classes
            # target_classes: b, num_queries * group_detr 所有query类别全部初始为no-object类型
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = (
                target_classes_o  # 匹配到的query类别设置为真实类别  b, num_queries * group_detr
            )

            # b, num_queries * group_detr, num_classes + 1
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )

            # target_classes.unsqueeze(-1): b, num_queries * group_detr, 1
            # target_classes_onehot的最后一维度，将匹配的num_classes位置设置为1，其余为0，为one-hot向量
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        matched_targets: _MatchedTargets,
    ) -> dict[str, torch.Tensor]:

        assert "pred_boxes" in outputs
        """
        idx = (batch_idx, src_idx)
                - batch_idx = [0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
                - src_idx = [0,1,0,1,0,1,2,...] [样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
        """
        idx = matched_targets.source_indices
        # [(cx, cy, w, h), ...] 所有样本的匹配框坐标值
        target_boxes = matched_targets.boxes
        # src_boxes: M, 4  其中 M = batch 内所有匹配对总数 = group_detr * sum(每个样本匹配框数)
        src_boxes = outputs["pred_boxes"][idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - elementwise_generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        matched_targets: _MatchedTargets | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_map: dict[str, _LossFunction] = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_boxes, matched_targets)
