import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.config import AppConfig

_FOCAL_LOSS_GAMMA = 2.0


class HungarianMatcher(nn.Module):
    """
    基于匈牙利算法（linear_sum_assignment）的预测框与真实框一对一匹配器。

    代价矩阵（cost matrix）是匹配的核心：矩阵第 i 行、第 j 列的元素
    cost_matrix[i, j] 表示「把第 i 个预测 query 分配给第 j 个真实目标」的总代价，
    代价越小越优先配对。总代价由三项加权求和得到（形状均为
    [num_queries * group_detr, b*N]，行=预测 query，列=真实目标）::

        cost_matrix = cost_bbox * L1(pred_box, tgt_box)
                    + cost_class * focal_class_cost(pred_logit @ tgt_class)
                    + cost_giou  * (-GIoU(pred_box, tgt_box))

    其中：
    - cost_bbox：预测框与真实框在 cxcywh 下的 L1 距离（torch.cdist, p=1）。
    - cost_class：focal loss 分类代价（见 _focal_classification_cost），
      取每个 query 对「目标真实类别」的净 focal 代价，见 forward 中 tgt_logits。
    - cost_giou：GIoU 取负（匹配目标是最小化总代价，故 IoU 越高代价越低）。

    由于代价矩阵是把 batch 内所有目标拼到最后一维算的，跨样本的列对当前样本
    无意义，因此会按 target_offsets 把矩阵切成每个样本独立的
    [num_queries * group_detr, N_i] 子矩阵，再各自送入匈牙利算法。

    当 group_detr > 1 时，query 被分成多组，每组在「同一批目标」上独立匹配，
    最后把各组的 (prediction_indices, target_indices) 拼接成该样本的最终配对。
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    def _focal_classification_cost(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算分类代价，使用focal loss
        参考：https://arxiv.org/pdf/1708.02002.pdf
        FL(p)=-α (1-p)^γ (log p)
        """
        # neg_cost_class = (1 - alpha) * (tgt_prob ** gamma) * (-(1 - tgt_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - tgt_prob) ** gamma) * (-(tgt_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        alpha = self.focal_alpha
        gamma = _FOCAL_LOSS_GAMMA
        probabilities = logits.sigmoid()
        # negative_cost = (1-α) p^γ (-log(1-p))
        negative_cost = (1 - alpha) * (probabilities**gamma) * (-F.logsigmoid(-logits))
        # positive_cost = α (1-p)^γ (-log p)
        positive_cost = alpha * ((1 - probabilities) ** gamma) * (-F.logsigmoid(logits))
        # !当 p 很大（预测 confidently 是目标类）正类代价≈0，负类代价很大 , cost ≈ 0 - 很大 = 很负 -> 强偏好这个匹配
        # !当 p 很小：正类代价很大,负类代价≈0，cost ≈ 正 → 回避这个匹配
        cost: torch.Tensor = positive_cost - negative_cost
        # 输出形状保持不变
        return cost

    @staticmethod
    def _assign_compact_cost_matrix(
        cost_matrix: torch.Tensor,  # num_queries * group_detr, b*N
        sizes: list[int],
        target_offsets: list[int],
        group_detr: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        num_queries = cost_matrix.shape[0]
        if num_queries % group_detr != 0:
            raise ValueError(
                f"num_queries ({num_queries}) 必须可以整除 group_detr ({group_detr})"
            )

        # 每组有多少个query
        group_num_queries = num_queries // group_detr
        indices = []

        for group_index in range(group_detr):
            # 每组开始query的索引
            group_start = group_index * group_num_queries
            # 得到每组的代价矩阵：num_queries, b*N
            grouped_cost_matrix = cost_matrix[
                group_start : group_start + group_num_queries
            ]

            """
            匈牙利算法返回的行列索引，行索引是query的索引，列索引是真实框的索引
            prediction_indices = [0, 1]     ← 行：query 0、query 1
            target_indices     = [0, 1]     ← 列：真实框 0、真实框 1
            匹配结果：
            query 0 ↔ 目标 0（代价 0.4）
            query 1 ↔ 目标 1（代价 0.3）
            query 2 不匹配（成为 no-object/背景）

            当前组每个样本的匈牙利算法的结果
            [
                
                (prediction_indices, target_indices), 第一个样本
                ...
            ]
            """
            group_indices = [
                linear_sum_assignment(
                    grouped_cost_matrix[
                        :, target_offsets[index] : target_offsets[index + 1]
                    ]
                )
                # 按照样本切割得到每个样本的代价矩阵
                for index in range(len(sizes))
            ]

            if group_index == 0:
                indices = group_indices
            else:
                indices = [
                    (
                        # 因为 group_indices 中的prediction_indices 为局部行号，需要加上该组的起始行号 group_start
                        # current[0] + group_num_queries * group_index 加上该组的起始行号 group_start，把局部行号还原成全局 query 索引
                        np.concatenate(
                            [previous[0], current[0] + group_num_queries * group_index]
                        ),
                        np.concatenate([previous[1], current[1]]),
                    )
                    # !当前组每个样本的的prediction_indices 拼接、target_indices 拼接
                    for previous, current in zip(indices, group_indices)
                ]
        """
        [
            # 第一个样本
            (
                prediction_indices, 每个组的prediction_indices 拼接在一起
                target_indices,  每个组的target_indices 拼接
            ),
            ...
        ]
        """
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        group_detr: int = 1,
    ):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # b * num_queries * group_detr, num_classes
        out_logits = outputs["pred_logits"].flatten(0, 1)
        # b * num_queries * group_detr, 4 （cx, cy, w, h）
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # list[(N,) ,...] -> b*N,
        tgt_ids = torch.cat([target["labels"] for target in targets])
        # list[(N, 4), ...] -> b*N, 4
        tgt_bbox = torch.cat([target["boxes"] for target in targets])

        # !计算giou，得到queries和真实框的两两配对的giou二维矩阵
        #  b * num_queries * group_detr, B*N
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        # giou代价，要选的是匹配代价最小的，所以取反
        cost_giou = -giou

        """
        # !数组索引取值，找到queries和这批真实框类别概率组成二维矩阵
         out_logits = [
           第一个query [0.12, 0.23, 0.24, 0.55, 0.68]
           第二个query [0.23, 0.11, 0.87, 0.44, 0.62]
         ]   
         比如说这批的所有真实框类别为tgt_ids =[0, 2, 3]

         tgt_logits=[
            第一个query [0.12, 0.24, 0.55]
            第二个query [0.23, 0.87, 0.44]
         ]
        """
        tgt_logits = out_logits[:, tgt_ids]  # b * num_queries * group_detr, b*N
        # 使用focal loss计算代价
        cost_class = self._focal_classification_cost(
            tgt_logits
        )  # b * num_queries * group_detr, b*N

        """
            !计算1范数（L1距离）
            x1：形状 [m, d]
            x2：形状 [n, d]
            p：距离范数（p=1 曼哈顿距离，p=2 欧氏距离，默认 2）
            返回：[m, n]，其中 result[i][j] = x1 第 i 行 与 x2 第 j 行的距离
        """
        cost_bbox = torch.cdist(
            out_bbox,
            tgt_bbox,
            p=1,  #
        )  # b * num_queries * group_detr, b*N

        """
        # !计算总代价，3个代价形状都是一样的，直接加权求和 b * num_queries * group_detr, b*N
        代价矩阵
                   真实框1  真实框2  真实框3   ...
        预测query1  [d11    d12     d13   ...]
        预测query2  [d21    d22     d23   ...]
        预测query3  [d31    d32     d33   ...]
        """
        cost_matrix: torch.Tensor = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )

        # b, num_queries * group_detr, b*N
        cost_matrix = cost_matrix.view(bs, num_queries, -1).float()
        # [每个样本的真实框的个数, ...]
        target_sizes = [len(target["boxes"]) for target in targets]

        # [每个样本的真实框数索引位置, ...]
        target_offsets = [0]
        for size in target_sizes:
            target_offsets.append(target_offsets[-1] + size)

        # cost_matrix是每个样本对所有真实框的 target 算了代价，跨图的那些计算是多余
        # [(num_queries * group_detr, N),...] 也就是每个样本的query数量和这个样本的真实框的代价矩阵
        cost_matrix_list = [
            cost_matrix[i, :, target_offsets[i] : target_offsets[i + 1]]
            for i in range(bs)
        ]

        """
        因为每个样本的query数量是相同的，所以按照最后一维进行拼接
        !num_queries * group_detr, b*N （ 最后一维可以用 target_offsets 去切割得到每个样本的代价矩阵 ）
    
        样本0: 列 0:2 (sizes[0]=2)             样本1: 列 2:3 (sizes[1]=1)
            t0   t1                                 t2
        q0  a    b                             q0   g
        q1  c    d                             q1   h
        q2  e    f                             q2   i


                      t0(样本0)   t1(样本0)   t2(样本1)
        q0(行0)          a         b         g
        q1(行1)          c         d         h
        q2(行2)          e         f         i
        """
        diagonal_cost_matrix = torch.cat(
            cost_matrix_list, dim=-1
        ).cpu()  # 匈牙利求解在cpu上计算
        return self._assign_compact_cost_matrix(
            diagonal_cost_matrix,
            sizes=target_sizes,
            target_offsets=target_offsets,
            group_detr=group_detr,
        )


def build_matcher(config: AppConfig) -> HungarianMatcher:
    common_kwargs = {
        "cost_class": config.train_config.cost_class,
        "cost_bbox": config.train_config.cost_bbox,
        "cost_giou": config.train_config.cost_giou,
        "focal_alpha": config.train_config.focal_alpha,
    }

    return HungarianMatcher(**common_kwargs)


if __name__ == "__main__":
    # ===== 与 LWDETR.forward 输出结构对齐的假数据（batch 位于第 0 维） =====
    # 说明：LWDETR 用 sigmoid 多分类，pred_logits 不含独立背景类；
    #       group_detr 默认 1，故查询数 = num_queries * group_detr = num_queries。
    bs, num_queries, num_classes = 2, 10, 9

    outputs = {
        # pred_logits: [b, num_queries * group_detr, num_classes]，sigmoid 前景分类 logits（无背景类）
        "pred_logits": torch.randn(bs, num_queries, num_classes),  # [2, 10, 9]
        # pred_boxes: [b, num_queries * group_detr, 4]，归一化 (cx, cy, w, h)
        "pred_boxes": torch.rand(bs, num_queries, 4),  # [2, 10, 4]
        # aux_outputs: 每个解码器中间层的输出列表（结构同主输出，去掉最后一层）
        "aux_outputs": [
            {
                "pred_logits": torch.randn(bs, num_queries, num_classes),  # [2, 10, 9]
                "pred_boxes": torch.rand(bs, num_queries, 4),  # [2, 10, 4]
            }
            for _ in range(2)  # 假设 2 个解码器中间层
        ],
    }

    # targets: list，每个元素对应一张图，含该图真实目标（与 matcher.forward 的 targets: list[dict] 一致）
    targets = [
        {  # 第 1 张图：3 个真实目标
            "labels": torch.randint(
                0, num_classes, (3,)
            ),  # [3] 前景类别 id ∈ [0, num_classes-1]
            "boxes": torch.rand(3, 4),  # [3, 4] 归一化 (cx, cy, w, h)
        },
        {  # 第 2 张图：5 个真实目标
            "labels": torch.randint(0, num_classes, (5,)),  # [5]
            "boxes": torch.rand(5, 4),  # [5, 4]
        },
    ]
    matcher = HungarianMatcher()
    matcher(outputs, targets, group_detr=1)
