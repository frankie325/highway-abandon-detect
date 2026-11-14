import torch
from torchvision.ops.boxes import box_area


def _assert_equal_length(boxes1: torch.Tensor, boxes2: torch.Tensor) -> None:
    """Reject unequal-length operands so a length-1 side cannot silently broadcast."""
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            "elementwise box ops expect boxes1 and boxes2 to have the same length, "
            f"got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    """
    即boxes1和boxes2中的每个box两两配对，形成一个 [N, M, 2] 的矩阵，
    boxes1[:, None, :2]:   [N, 1, 2]
    boxes2[:, :2]:         [M, 2]  → 广播时自动补为 [1, M, 2]
    ─────────────────────────────
    广播结果:              [N, M, 2]
    """
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(
        min=0
    )  # [N,M,2] #求出 dx * dy，并把张量里所有小于 0 的值截断为 0，保证宽高最小是 0（零宽/零高框），面积不会变负
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def elementwise_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU and union for pre-matched box pairs.

    Unlike ``box_iou``, this avoids materializing the full NxN pairwise matrix
    (of which only the diagonal is used for matched pairs), so peak memory is
    O(N) instead of O(N^2). ``boxes1`` and ``boxes2`` must have the same length
    and be in [x0, y0, x1, y1] format; a length mismatch raises ``ValueError``
    rather than silently broadcasting a length-1 operand.

    The numerics (op order and the ``eps=1e-7`` union clamp) are identical to the
    diagonal of ``box_iou``, so results carry no dtype dependence beyond that of
    the pairwise version under FP16/FP32.

    Returns:
        iou: the [N] tensor of IoU values for each matched pair.
        union: the [N] tensor of union areas for each matched pair.
    """
    _assert_equal_length(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    eps = 1e-7
    # Clamp only the degenerate (union==0) case so identical non-degenerate boxes
    # yield IoU==1.0 exactly; adding eps unconditionally would break that identity.
    iou = inter / union.clamp(min=eps)
    return iou, union


def elementwise_generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/ for pre-matched box pairs.

    Equivalent to the diagonal of ``generalized_box_iou`` but without building the
    NxN matrix, giving O(N) instead of O(N^2) peak memory. The boxes should be in
    [x0, y0, x1, y1] format, and ``boxes1``/``boxes2`` must have the same length; a
    length mismatch raises ``ValueError`` rather than silently broadcasting.

    Returns a [N] tensor, one GIoU value per matched pair.
    """
    _assert_equal_length(boxes1, boxes2)
    # Degenerate (zero-area) boxes would divide 0/0; eps in the denominators keeps results finite.
    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    eps = 1e-7
    # Clamp only when enclosing area is zero (degenerate enclosing box) so normal boxes remain exact.
    return iou - (area - union) / area.clamp(min=eps)
