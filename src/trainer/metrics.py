"""检测评估输入、混淆矩阵及验证图表。"""

from io import BytesIO

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve
from torchvision.ops import box_iou

from src.utils.box_ops import box_cxcywh_to_xyxy
from src.utils.tensors import NestedTensor


def prepare_detection_inputs(
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    outputs: dict[str, torch.Tensor],
    *,
    max_detections: int = 100,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """把模型预测和真实标注整理为目标检测指标所需的逐图结果。

    流程：先根据 padding mask 找到每张图片的有效区域；然后把分类分数
    转成概率，选取最高分的预测并定位对应的 Query 和类别；最后将预测框
    和真实框从归一化 cxcywh 转为有效图片区域中的像素 xyxy，并过滤退化框。

    返回的 predictions 和 ground_truths 与 batch 中的图片顺序一致，供 mAP、
    PR 曲线和混淆矩阵使用。这里只保留最高分的 max_detections 个候选，不按
    固定置信度过滤；混淆矩阵需要的置信度筛选在后续步骤单独进行。
    """
    # B：批次大小；Q：Query 数；C：类别数；N_i：第 i 张图的真实目标数。
    # 输入 outputs["pred_logits"]：[B, Q, C]；outputs["pred_boxes"]：[B, Q, 4]。
    # targets 是长度为 B 的列表，第 i 项的 boxes：[N_i, 4]，labels：[N_i]。
    if samples.mask is None:
        raise ValueError("检测评估需要 padding mask 确定每张图片的有效尺寸")
    # 两个列表最终均包含 B 个字典，每张图片分别保留数量可变的预测与标注。
    predictions, ground_truths = [], []
    for index, target in enumerate(targets):
        # samples.mask：[B, H_pad, W_pad] → 单张 mask：[H_pad, W_pad]。
        mask = samples.mask[index].detach().cpu()
        # 第一列 [H_pad]、第一行 [W_pad] → 有效高度和宽度，均为标量。
        height, width = int((~mask[:, 0]).sum()), int((~mask[0, :]).sum())
        if height <= 0 or width <= 0:
            raise ValueError("验证图片没有有效像素区域")
        # scale：[4]，与 xyxy 坐标逐元素相乘时自动广播。
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)

        def pixel_boxes(boxes: torch.Tensor) -> torch.Tensor:
            # 输入 [T, 4] 归一化 cxcywh → 输出 [T, 4] 像素 xyxy；空框为 [0, 4]。
            xyxy = box_cxcywh_to_xyxy(boxes.detach().float().cpu()) * scale
            xyxy[:, 0::2].clamp_(0, width)   # [T, 2]：裁剪 x1、x2。
            xyxy[:, 1::2].clamp_(0, height)  # [T, 2]：裁剪 y1、y2。
            return xyxy                       # [T, 4]

        # 单张图 logits：[Q, C] → sigmoid 概率 probs：[Q, C]。
        probs = outputs["pred_logits"][index].detach().float().cpu().sigmoid()
        flat = probs.flatten()  # [Q, C] → [Q*C]。
        # K=min(max_detections, Q*C)；最高分 scores 和展平索引 selected 均为 [K]。
        scores, selected = flat.topk(min(max_detections, flat.numel()))
        # 展平索引 [K] → Query 索引 [K]；例如 C=3、索引 7 对应 Query 2。
        queries = torch.div(selected, probs.shape[-1], rounding_mode="floor")
        labels = (selected % probs.shape[-1]).long()  # 展平索引 [K] → 类别编号 [K]。
        # 单图预测框 [Q, 4] → 按 Query 选取 [K, 4] → 像素 xyxy [K, 4]。
        boxes = pixel_boxes(
            outputs["pred_boxes"][index][queries.to(outputs["pred_boxes"].device)]
        )
        # 预测框 [K, 4] → 有效框布尔掩码 [K]；裁剪后宽或高为 0 的框被过滤。
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        # 过滤后剩 K_i 个预测：boxes [K_i, 4]，scores / labels 均为 [K_i]。
        predictions.append(
            {"boxes": boxes[valid], "scores": scores[valid], "labels": labels[valid]}
        )

        boxes = pixel_boxes(target["boxes"])  # 真实框 [N_i, 4] → 像素 xyxy [N_i, 4]。
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])  # [N_i]。
        # 过滤后剩 M_i 个目标：boxes [M_i, 4]，labels [M_i]；空目标也保持二维框形状。
        ground_truths.append(
            {
                "boxes": boxes[valid],
                "labels": target["labels"].detach().cpu().long()[valid],
            }
        )
    return predictions, ground_truths  # 两个长度为 B 的逐图字典列表。


class DetectionConfusionMatrix(Metric):
    """累积检测混淆矩阵：行是真实类别、列是预测类别。

    C 个类别对应 [C+1, C+1] 矩阵，额外的索引 C 表示虚拟背景：
    最后一行统计误检，最后一列统计漏检；背景与背景的交叉位置不计真负例。
    """

    full_state_update = False

    def __init__(self, num_classes: int, score_threshold: float, iou_threshold: float):
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        # 多设备运行时对各设备的计数求和；矩阵不计背景真负例。
        self.add_state(
            "matrix",
            default=torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        """处理一个 batch：逐图按分数与 IoU 匹配，并把结果累加到矩阵。

        以车辆和行人两个类别为例，假设置信度阈值为 0.3、IoU 阈值为 0.5：
        - TP（正确检测）：图片里有一辆真实车辆，预测框 A 的分数为 0.9，
          与该车辆的 IoU 为 0.8、预测类别也是车辆；A 匹配成功，矩阵[车辆, 车辆] +1。
        - FP（误检）：在上述图片里，另一个车辆预测框 B 的分数为 0.8，与同一车辆的
          IoU 为 0.7。A 先匹配该目标后，B 不能重复匹配，矩阵[背景, 车辆] +1。
          如果预测框没有达到 IoU 阈值的未匹配目标，也会按同样方式记作 FP。
        - FN（漏检）：另一张图中有一个真实行人，但没有预测框成功匹配它，
          则矩阵[行人, 背景] +1。如果预测分数低于 0.3，该预测先被过滤，
          不会单独记 FP；真实目标若最终仍未匹配，则记作 FN。
        - 类别错误：真实行人与预测车辆框成功匹配时，矩阵[行人, 车辆] +1；
          按类别统计时，它对行人类别是一次漏检，对车辆类别是一次误检。

        每个预测框和真实框最多参与一次匹配，避免同一目标被重复算作正确检测。

        本方法按预测分数从高到低进行贪心匹配，不是训练时的 Hungarian 匹配。
        匹配时不限制类别：位置匹配但类别错误会记入混淆矩阵的非对角线。
        """
        if len(predictions) != len(targets):
            raise ValueError("预测和标注图片数量不一致")
        bg = self.num_classes
        for pred, target in zip(predictions, targets, strict=True):
            # 每张图预测框 [K, 4]，分数和标签 [K]；keep 为 [K] 的布尔掩码。
            # 置信度阈值只在混淆矩阵使用，前面的 mAP 输入不会在这里被原地修改。
            keep = pred["scores"] >= self.score_threshold
            boxes, labels, scores = (
                pred["boxes"][keep],
                pred["labels"][keep],
                pred["scores"][keep],
            )
            # 筛选后剩 P 个预测框，真实框数量为 G：
            # boxes：[P, 4]、labels/scores：[P]；gt_boxes：[G, 4]、gt_labels：[G]。
            gt_boxes, gt_labels = target["boxes"], target["labels"]
            if ((labels < 0) | (labels >= bg)).any() or (
                (gt_labels < 0) | (gt_labels >= bg)
            ).any():
                raise ValueError("类别编号超出混淆矩阵范围")
            # ious：[P, G]，第 i 行第 j 列是第 i 个预测框与第 j 个真实框的 IoU。
            # 如果任一方为空，不构建 IoU 矩阵，剩余框按误检或漏检处理。
            ious = box_iou(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else None
            # matched 保存已经匹配过的真实框下标，保证一个真实框最多匹配一次。
            # 每个预测框也只会被遍历一次，因此最多匹配一个真实框。
            matched: set[int] = set()
            # 按置信度由高到低做一对一 IoU 匹配；不预先过滤类别以统计类别混淆。
            for pred_idx in scores.argsort(descending=True).tolist():
                candidates = [idx for idx in range(len(gt_boxes)) if idx not in matched]
                gt_idx = None
                if candidates and ious is not None:
                    # 从尚未匹配的真实框中选择 IoU 最大的一个，达到阈值才算匹配。
                    best = int(ious[pred_idx, candidates].argmax())
                    candidate = candidates[best]
                    if float(ious[pred_idx, candidate]) >= self.iou_threshold:
                        gt_idx = candidate
                if gt_idx is None:
                    # 没有匹配到真实框：真实背景 → 预测类别，即误检 FP。
                    self.matrix[bg, int(labels[pred_idx])] += 1
                else:
                    # 匹配成功：真实类别 → 预测类别；类别相同记在对角线上。
                    matched.add(gt_idx)
                    self.matrix[int(gt_labels[gt_idx]), int(labels[pred_idx])] += 1
            for gt_idx, label in enumerate(gt_labels.tolist()):
                if gt_idx not in matched:
                    # 未匹配的真实框：真实类别 → 预测背景，即漏检 FN。
                    self.matrix[label, bg] += 1

    def compute(self) -> torch.Tensor:
        return self.matrix


def _as_image(figure: Figure) -> Image.Image:
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    output.seek(0)
    with Image.open(output) as image:
        return image.convert("RGB").copy()


def draw_pr_curve(metrics: dict) -> Image.Image:
    """用 TorchMetrics 绘制 COCO IoU=0.5、全部面积下的逐类别 PR 曲线。

    COCO 已经完成检测框匹配并计算 101 个召回率上的插值精度；这里只借用
    分类 PR 指标的 plot 接口展示这些点，不重新按分类任务计算 PR 或 AUC。
    """
    figure = Figure(figsize=(7, 5), layout="constrained")
    ax = figure.subplots()
    precision = (
        metrics["precision"].detach().cpu()
    )  # [IoU, recall, class, area, maxDets]
    classes = metrics["classes"].detach().cpu().reshape(-1).tolist()
    recall = torch.linspace(0, 1, precision.shape[1])
    precisions, recalls, labels = [], [], []
    for index, label in enumerate(classes):
        values = precision[0, :, index, 0, -1]
        valid = values >= 0
        if valid.any():
            precisions.append(values[valid])
            recalls.append(recall[valid])
            labels.append(f"Class {label}")
    if precisions:
        # 多分类指标要求至少两个类别；此处仅使用 plot，单类别时设置为 2
        # 不会增加曲线，也不会重新计算指标。曲线数量由传入的数据决定。
        plotter = MulticlassPrecisionRecallCurve(num_classes=max(2, len(precisions)))
        # COCO 曲线按召回率采样，绘图不使用阈值；第三项仅满足接口格式。
        thresholds = [torch.empty(0) for _ in precisions]
        plotter.plot(curve=(precisions, recalls, thresholds), ax=ax)
        # TorchMetrics 默认用列表下标命名曲线；改成实际的 COCO 类别编号。
        for line, label in zip(ax.lines, labels, strict=True):
            line.set_label(label)
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No ground-truth objects",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        title="Precision-Recall (IoU = 0.50)",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.3)
    return _as_image(figure)


def draw_confusion_matrix(
    matrix: torch.Tensor, *, score_threshold: float, iou_threshold: float
) -> Image.Image:
    """绘制检测混淆矩阵；右下角背景/背景不表示真负例。"""
    values = matrix.detach().cpu()
    labels = [f"Class {idx}" for idx in range(len(values) - 1)] + ["Background"]
    figure = Figure(figsize=(7, 6), layout="constrained")
    ax = figure.subplots()
    # 先由检测指标计算矩阵，分类指标仅负责绘制，不参与框匹配或重新计数。
    plotter = MulticlassConfusionMatrix(num_classes=len(values))
    plotter.plot(val=values, ax=ax, labels=labels, cmap="Blues")
    figure.colorbar(ax.images[0], ax=ax)
    ax.set(
        xlabel="Predicted",
        ylabel="Ground truth",
        title=f"Confusion matrix (score ≥ {score_threshold:g}, IoU ≥ {iou_threshold:g})",
    )
    # 检测任务没有可枚举的背景真负例；将右下角的默认数值改成破折号。
    background_position = (len(values) - 1, len(values) - 1)
    for text in ax.texts:
        if text.get_position() == background_position:
            text.set_text("—")
    return _as_image(figure)
