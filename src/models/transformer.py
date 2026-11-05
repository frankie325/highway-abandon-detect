from collections.abc import Callable
from torch._tensor import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from loguru import logger
from collections.abc import Sequence

from .math import MLP
from .ms_deform_attn import MSDeformAttn
from src.config import AppConfig


"""
候选框的几何 (x, y, w, h)进行位置编码
目的是把 num_queries * group_detr 个 proposal 的框信息，编码成 decoder 的初始 query
而是由图像内容驱动生成、且携带明确几何先验的
每个 query 一上来就知道"我负责 (x,y,w,h) 这个框"。所以两阶段收敛更快、小目标更好
"""


def gen_sineembed_for_position(
    pos_tensor: torch.Tensor, dim: int = 128
) -> torch.Tensor:
    # b, num_queries * group_detr, 4 = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")

    """
    sin/cos 交错后展平
    [ x 的 128 维 | y 的 128 维 | w 的 128 维 | h 的 128 维 ]
        ↑ 每 128 维内部是 sin(θ₀),cos(θ₀),sin(θ₁),cos(θ₁),... 交错
    """
    return pos  # b, num_queries * group_detr, 4 * dim = 2*d_model


def gen_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    spatial_shapes: Sequence[tuple[int, int]],
    unsigmoid: bool = True,
):
    r"""
    为每个特征层级（多尺度）的每个空间位置生成用于解码器的初始候选框（proposals），
    并将编码器输出 memory 处理为解码器可用的 output_memory。

    这是检测用 Transformer 中标准的 "gen_encoder_output_proposals"：依据
    memory_padding_mask 计算各特征层真实有效区域的高宽，用网格坐标为每个位置生成
    归一化中心 (cx, cy)，并以 0.05 * (2**level) 作为默认宽高 (w, h)，形成粗到细的
    多尺度参考框先验，供解码器 query 的初始化/参考点使用。

    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    batch_size, _, _ = memory.shape
    current = 0  # 用来从序列中取出特征层
    proposals = []
    for lvl, (height, width) in enumerate(spatial_shapes):

        # 取出当前层特征图的mask，True = 该位置是 padding（无效），False = 真实有效内容
        mask_flatten = memory_padding_mask[
            :, current : current + height * width
        ]  # b, h*w
        mask_flatten = mask_flatten.reshape(batch_size, height, width)  # b, h, w

        # 计算图像区域的高度、宽度 (b,)
        valid_height = torch.sum(~mask_flatten[:, :, 0], 1)
        valid_width = torch.sum(~mask_flatten[:, 0, :], 1)

        # 为每个位置生成中心坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0,
                height - 1,
                height,
                dtype=torch.float32,
                device=memory.device,
            ),  # [0, 1, 2, ..., height-1]
            torch.linspace(
                0,
                width - 1,
                width,
                dtype=torch.float32,
                device=memory.device,
            ),  # [0, 1, 2, ..., width-1]
            indexing="ij",
        )
        # grid_y.shape = h, w
        # grid_x.shape = h, w
        # 构建成网格坐标h, w, 2
        grid = torch.stack([grid_x, grid_y], dim=-1)  # h, w, 2(保存的是x,y坐标)
        # b, 2 -> b, 1, 1, 2
        scale = torch.stack([valid_width, valid_height], dim=-1).reshape(-1, 1, 1, 2)
        grid = (
            grid.unsqueeze(0) + 0.5
        )  # grid.shape =  1, h, w, 2 下面和scale相处时可以进行广播。+0.5表示把整数下标变成像素中心坐标（如 0→0.5）

        # !坐标为相对于图像区域的相对坐标，和生成参考点时一样，padding区域相对坐标可能会大于1，所以下面使用> 0.01和< 0.99来过滤掉padding区域的坐标
        proposals_grid = grid / scale.float()  # b, h, w, 2

        # !不同 level 默认框大小不同——越粗的层级（level 越大、特征图越小）默认框越大，形成多尺度先验
        wh = (
            torch.ones_like(proposals_grid) * 0.05 * (2.0**lvl)
        )  # b, h, w, 2(最后一维w,h表示预测框的宽高)

        proposal = torch.cat(
            [proposals_grid, wh], -1
        )  # b, h, w, 4 (x,y,w,h 中心点坐标 和 预测框的宽高)

        proposals.append(proposal.reshape(batch_size, -1, 4))  # b, h*w, 4
        current += height * width

    output_proposals = torch.cat(proposals, dim=1)  # b, \sum{h*w}, 4
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(
        -1, keepdim=True
    )  # b, \sum{h*w}, 1 （每个空间位置的 4 个坐标是否都落在 (0.01, 0.99) 区间内，是则该位置为有效候选框）

    # 是否反sigmoid
    if unsigmoid:
        # 把编码器初始框从归一化 [0,1] 预转换到logits实数空间，
        # 以便解码器第一层就能"在 logit 空间加 delta"做稳定、不饱和的框精修，同时让 padding/越界框能用 inf 干净地标记为无效
        output_proposals = torch.log(
            output_proposals / (1 - output_proposals)
        )  # b, \sum{h*w}, 4

        # memory_padding_mask.unsqueeze(-1) -> b, \sum{h*w}, 1 -> 广播
        # !凡属 padding 的 proposal，坐标全设成 inf，标记为无效框
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )  # b, \sum{h*w}, 4
        # !区间外位置也标记为无效框
        output_proposals: Tensor = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )  # b, \sum{h*w}, 4
    else:
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )  # b, \sum{h*w}, 4

        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float(0)
        )  # b, \sum{h*w}, 4

    # !同理：memory位置也需要mask，padding位置的memory也设置为0
    output_memory = memory  # b, \sum{h*w}, d_model
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0)
    )  # b, \sum{h*w}, d_model

    output_memory = output_memory.masked_fill(
        ~output_proposals_valid, float(0)
    )  # b, \sum{h*w}, d_model

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        sa_nhead: int = 8,
        ca_nhead: int = 8,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        num_feature_levels: int = 4,
        bbox_reparam: bool = False,
    ):
        super().__init__()
        self.encoder = None
        self.enc_out_class_embed: nn.ModuleList | None = None
        self.enc_out_bbox_embed: nn.ModuleList | None = None

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.group_detr = group_detr
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam

        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            # normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            # dec_n_points=dec_n_points,
            skip_self_attn=False,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            # decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            # lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
            # enable_keypoint_processing=use_grouppose_keypoints,
            # num_keypoints_per_class=self.num_keypoints_per_class,
            # grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
        )

        if self.two_stage:
            # 分组
            self.enc_output = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(group_detr)]
            )
            self.enc_output_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(group_detr)]
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    """
    会计算每个特征层中图像区域的比例
    valid_ratio_w = 图像宽度 / 总宽度
    valid_ratio_h = 图像高度 / 总高度
    """

    def get_valid_ratio(self, mask: torch.Tensor) -> torch.Tensor:
        _, height, width = mask.shape  # b, h, w
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height.float() / height
        valid_ratio_w = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio  # b, 2

    def forward(
        self,
        srcs: list[torch.Tensor],
        masks: list[torch.Tensor],
        pos_embeds: list[torch.Tensor],
        refpoint_embed: torch.Tensor,
        query_feat: torch.Tensor,
    ):
        src_flatten = []
        mask_flatten_parts: list[torch.Tensor] = []
        lvl_pos_embed_flatten_parts = []
        spatial_shapes_hw: list[tuple[int, int]] = []

        # 遍历所有特征图
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            b, c, h, w = src.shape
            spatial_shapes_hw.append((h, w))

            src = src.flatten(2).transpose(1, 2)  # 将特征图展平为 b, h*w, d_model
            pos_embed = pos_embed.flatten(2).transpose(
                1, 2
            )  # 将位置编码展平为 b, h*w, d_model
            lvl_pos_embed_flatten_parts.append(pos_embed)
            src_flatten.append(src)
            mask = masks[lvl].flatten(1)  # b, h*w
            mask_flatten_parts.append(mask)

        memory = torch.cat(
            src_flatten, dim=1
        )  # 将特征图沿着h*w方向进行拼接： b, \sum{hxw}, d_model
        mask_flatten = torch.cat(mask_flatten_parts, dim=1)  # b, \sum{hxw}
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1
        )  # b, 特征层层数, 2

        lvl_pos_embed_flatten = torch.cat(
            lvl_pos_embed_flatten_parts, dim=1
        )  # b, \sum{hxw}, d_model

        """
        spatial_shapes: 4个特征图的尺寸
        假设为：
        spatial_shapes = [
            (28, 40),
            (14, 20),
            (7, 10),
            (4, 5),
        ]
        """
        spatial_shapes = torch.as_tensor(
            spatial_shapes_hw, dtype=torch.long, device=src_flatten[0].device
        )

        """
            spatial_shapes.new_zeros((1, )) = [0] 输出一个为0的标量
            spatial_shapes.prod(1) = [28*40, 14*20, 7*10, 4*5] 按照第1维计算乘积
            .cumsum(0)累加求和，[1120, 1400, 1470, 1490]
            [:-1] 去掉最后一个 [1120, 1400, 1470]
            torch.cat(([0], [1120, 1400, 1470])) = [0, 1120, 1400, 1470]
            !level_start_index的作用就是记住各层特征图在这个长序列中的起始位置
        """
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        logger.debug("当前批次的memory形状：{}", memory.shape)
        logger.debug("当前批次的mask_flatten形状：{}", mask_flatten.shape)
        logger.debug("当前批次的spatial_shapes形状：{}", spatial_shapes)
        logger.debug("当前批次的level_start_index：{}", level_start_index)

        # !两阶段，由Encoder预测、精修候选框
        if self.two_stage:
            assert self.enc_out_class_embed is not None
            assert self.enc_out_bbox_embed is not None

            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes_hw, unsigmoid=not self.bbox_reparam
            )

            """
            标准 DETR 用一对一二分匹配：num_queries 个查询（如 300）和图像里的真实框做最优匹配。早期训练时：
                查询太少、真实框相对多，匹配很"拥挤"且不稳定——很多查询争抢同一简单目标，容易陷入退化匹配；
                一对一意味着每个真实框只"监督"到 1 个查询，监督信号稀疏，要几百个 epoch 才收敛。
            把 num_queries 个查询复制成 group_detr 组、每组独立和真实框做一对一匹配，
            相当于让每个真实框同时被多组查询"抢着学"，匹配更稳、梯度更密；推理时只留 1 组，所以加速训练却不增加推理成本
            """
            refpoint_embed_ts_parts, memory_ts_parts, boxes_ts_parts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                # b, \sum{h*w}, d_model
                output_memory_gidx = self.enc_output_norm[g_idx](
                    self.enc_output[g_idx](output_memory)
                )
                # 分类预测：b, \sum{h*w}, num_classes
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                    output_memory_gidx
                )

                # 选择排名前topk的框，min限制queries的数量不能超过\sum{h*w}
                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[1])

                # 取num_classes中最大的值，即为预测的类别 b, \sum{h*w}
                values, _ = enc_outputs_class_unselected_gidx.max(-1)

                # 用来取置信度前topk个框的索引
                _, topk_proposals_gidx = torch.topk(values, topk, dim=1)  # b, topk

                # !得到前topk个queries: b, topk, d_model
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx,
                    1,
                    topk_proposals_gidx.unsqueeze(-1).expand(-1, -1, self.d_model),
                )

                # !得到前topk个bbox: b, topk, 4
                output_proposals_gidx = torch.gather(
                    output_proposals,
                    1,
                    topk_proposals_gidx.unsqueeze(-1).expand(-1, -1, 4),
                )
                logger.debug(
                    "第{}组，前topk个queries形状：{}", g_idx, tgt_undetach_gidx.shape
                )
                logger.debug(
                    "第{}组，前topk个bbox形状：{}",
                    g_idx,
                    output_proposals_gidx.shape,
                )

                """
                bbox_reparam: 边界框重参数化开关。开启时不再直接预测绝对坐标，
                而是预测相对初始 proposal（参考点）的偏移与 log 尺度缩放，再重参数化回
                (cx, cy, w, h)：中心 = 原中心 + 偏移 * 原宽高，宽高 = 原宽高 * exp(偏移)。
                这种"相对参考点回归"可让训练更稳定、收敛更快；关闭则使用绝对坐标预测。
                """
                if self.bbox_reparam:

                    # 得到ΔxΔywh的预测偏移量
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](
                        tgt_undetach_gidx
                    )

                    # 中心：cx,cy = 原中心 + 预测偏移 * 原宽高 :  b, topk, 2
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2]  # 预测偏移
                        * output_proposals_gidx[..., 2:]  # 原宽高
                        + output_proposals_gidx[..., :2]  # 原中心
                    )

                    # 宽高：新 wh = exp(预测值) * 原 wh
                    enc_outputs_coord_wh_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2].exp()
                        * output_proposals_gidx[..., 2:]
                    )

                    # !最终拼回 (cx, cy, w, h):  b, topk, 4
                    refpoint_embed_gidx_undetach = torch.cat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                    )
                else:
                    # !output_proposals_gidx在gen_encoder_output_proposals方法中已经做了反归一化，所以可以直接预测绝对坐标，
                    # b, topk, 4
                    refpoint_embed_gidx_undetach = (
                        self.enc_out_bbox_embed[g_idx](
                            tgt_undetach_gidx
                        )  # 直接预测绝对坐标
                        + output_proposals_gidx
                    )

                # detach分离梯度，作为解码器第一层的初始参考框
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
                refpoint_embed_ts_parts.append(refpoint_embed_gidx)

                # 作为解码器的queries
                memory_ts_parts.append(tgt_undetach_gidx)
                # 未 detach 的 boxes_ts_parts：保留梯度，用于两阶段编码器自身的框/分类监督（直接拿这些 proposal 和 GT 算 loss，逼编码器出好 proposal）
                boxes_ts_parts.append(refpoint_embed_gidx_undetach)

            # b, topk * group_detr, 4
            refpoint_embed_ts = torch.cat(refpoint_embed_ts_parts, dim=1)
            # b, topk * group_detr, d_model
            memory_ts = torch.cat(memory_ts_parts, dim=1)
            # b, topk * group_detr, 4
            boxes_ts = torch.cat(boxes_ts_parts, dim=1)
            logger.debug(
                "分组匹配后，前topk * group_detr个queries形状：{}", memory_ts.shape
            )
            logger.debug(
                "分组匹配后，前topk * group_detr个bbox形状：{}", boxes_ts.shape
            )
            logger.debug(
                "分组匹配后，前topk * group_detr个bbox的{}：{}",
                "归一化坐标" if self.bbox_reparam else "绝对坐标",
                refpoint_embed_ts,
            )

        if self.num_decoder_layers > 0:
            bs = memory.shape[0]

            # num_queries*group_detr, hidden_dim -> b, num_queries*group_detr, hidden_dim
            tgt = query_feat.unsqueeze(0).expand(bs, -1, -1).contiguous()
            """
            decoder 的初始参考锚框来自可学习的Embedding层
            锚框坐标为cx, cy, w, h
            """
            # num_queries*group_detr, 4 -> b, num_queries*group_detr, 4
            refpoint_embed = refpoint_embed.unsqueeze(0).expand(bs, -1, -1).contiguous()

            """
            两阶段采用混合初始化：
            Decoder 的 query 参考点一半由图像内容（Encoder proposal）引导、一半自由学习，两者拼接
                - proposal 引导的部分：让 query 一开始就聚焦在 encoder 粗检出的目标区域，收敛快、定位准。
                - 纯学习 anchor 部分：不依赖 encoder，能补覆盖 proposal 可能漏掉的、或难被粗检出的目标。
            既利用内容先验，又保留学习冗余
            """
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]  # ts_len = topk * group_detr

                # topk 一般来说就等于 num_queries，除非批次特征图\sum{hxw} <  num_queries, 此时topk = \sum{hxw} < num_queries
                refpoint_embed_ts_subset = refpoint_embed[
                    ..., :ts_len, :
                ]  # b, topk * group_detr, 4
                # topk = num_queries时，refpoint_embed_subset为空切片，topk < num_queries时， refpoint_embed_subset为纯可学习查询
                refpoint_embed_subset = refpoint_embed[
                    ..., ts_len:, :
                ]  # b, num_queries * group_detr - topk * group_detr, 4

                if self.bbox_reparam:
                    # !单阶段预测偏移初始时为0，后面经过解码器的修正，会更新为预测偏移
                    # 0*wh + 中心 = 中心，exp(0)*wh = wh

                    # 中心：cx,cy = 原中心 + 预测偏移 * 原宽高 :  b, topk, 2
                    refpoint_embed_cxcy = (
                        refpoint_embed_ts_subset[..., :2]  # 预测偏移
                        * refpoint_embed_ts[..., 2:]  # 原宽高（已被编码器框头精修过）
                        + refpoint_embed_ts[..., :2]  # 原中心（已被编码器框头精修过）
                    )

                    # 宽高： 新 wh = exp(预测值) * 原 wh
                    refpoint_embed_wh = (
                        refpoint_embed_ts_subset[..., 2:].exp()
                        * refpoint_embed_ts[..., 2:]  # 原宽高（已被编码器框头精修过）
                    )

                    # 拼回 (cx, cy, w, h) : b, topk * group_detr, 4
                    refpoint_embed_ts_subset = torch.cat(
                        [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                    )
                else:
                    # b, topk * group_detr, 4
                    refpoint_embed_ts_subset = (
                        refpoint_embed_ts_subset + refpoint_embed_ts  # 直接预测绝对坐标
                    )

                # b, num_queries * group_detr, 4
                refpoint_embed = torch.cat(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2
                )

            """
            概念来自《Vision Transformers Need Registers》(2024)
            标准 DETR 常有部分 query 退化成"不负责任何目标"的冗余槽位；
            显式加几个 register token 让这些"无用"角色有了固定去处，反而让 object query 更专注于目标
            """
            # if self.num_registers > 0:

            decoder_memory = memory

            logger.debug(
                "decoder 输入形状 -> memory:{}, tgt:{}, mask_flatten:{}, pos:{}, "
                "refpoints:{}, spatial_shapes:{}, level_start_index:{}, valid_ratios:{}, "
                "spatial_shapes_hw:{}",
                decoder_memory.shape,
                tgt.shape,
                mask_flatten.shape,
                lvl_pos_embed_flatten.shape,
                refpoint_embed.shape,
                spatial_shapes.shape,
                level_start_index.shape,
                valid_ratios.shape,
                spatial_shapes_hw,
            )

            decoder_outputs = self.decoder(
                tgt,  # b, num_queries * group_detr, d_model Embedding层权重，可学习的向量
                decoder_memory,  # b, \sum{hxw}, d_model
                memory_key_padding_mask=mask_flatten,  # b, \sum{hxw}
                pos=lvl_pos_embed_flatten,  # b, \sum{hxw}, d_model
                refpoints_unsigmoid=refpoint_embed,  # b, num_queries * group_detr, 4
                level_start_index=level_start_index,  # [0, ...]
                spatial_shapes=spatial_shapes,
                spatial_shapes_hw=spatial_shapes_hw,
                valid_ratios=valid_ratios,  # b, 特征层层数, 2
            )
            # hs: decoder_layers, b, num_queries * group_detr, d_model
            # references: decoder_layers, b, num_queries * group_detr, 4
            hs, references = decoder_outputs

            return_values = [hs, references]

            if self.two_stage:
                return_values.append(
                    memory_ts
                )  # memory_ts: b, topk * group_detr, d_model
                if self.bbox_reparam:
                    return_values.append(boxes_ts)  # boxed_ts:  b, topk * group_detr, 4
                else:
                    return_values.append(boxes_ts.sigmoid())
            else:
                return_values.extend([None, None])

        logger.debug("Transformer返回结果：隐藏层输出hs：{}", return_values[0].shape)
        logger.debug(
            "Transformer返回结果：参考点输出references：{}", return_values[1].shape
        )
        logger.debug(
            "Transformer返回结果：编码器预测的前topk个采样位置输出memory_ts：{}",
            return_values[2].shape,
        )
        logger.debug(
            "Transformer返回结果：编码器预测的前topk个锚框boxes_ts：{}",
            return_values[3].shape,
        )
        return tuple(return_values)


def _get_clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        sa_nhead: int,  # 自注意力机制头数
        ca_nhead: int,  # 交叉注意力机制头数
        dim_feedforward: int = 2048,  # 前馈层隐藏层维度数
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        group_detr: int = 1,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        skip_self_attn: bool = False,
        # enable_keypoint_processing: bool = False,
        # grouppose_keypoint_dim_downscale: int = 1,
        # keypoint_cross_attn: bool = True,
        # inter_instance_kp_attn: bool = False,
    ) -> None:
        super().__init__()

        self.group_detr = group_detr

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,  # 特征层数量
            n_heads=ca_nhead,  # 多头注意力头数
            n_points=dec_n_points,  # 采样点数
        )

        # 前馈层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(
        self, tensor: torch.Tensor, pos: torch.Tensor | None
    ) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        # tgt_mask: torch.Tensor | None = None,
        # memory_mask: torch.Tensor | None = None,
        # tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        layer_index: int = 0,
    ):
        # b, num_queries * group_detr, d_model
        bs, np, _ = tgt.shape

        logger.debug("===========解码器层第{}自注意力计算==============", layer_index)
        # b, num_queries * group_detr, d_model
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        if self.training:
            q_groups = q.split(
                np // self.group_detr, dim=1
            )  # 列表[(b, num_queries, d_model),...]
            q = torch.cat(q_groups, dim=0)  # b*group_detr, num_queries, d_model
            k = q  # b*group_detr, num_queries, d_model
            v_groups = tgt.split(np // self.group_detr, dim=1)
            v = torch.cat(v_groups, dim=0)  # b*group_detr, num_queries, d_model

        # !自注意力层不需要mask
        # b*group_detr, num_queries, d_model
        tgt2 = self.self_attn(
            q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False
        )[0]

        if self.training:
            # 再还原成 b, num_queries * group_detr, d_model
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)  # b, num_queries * group_detr, d_model
        logger.debug(
            "===========解码器第{}层自注意力计算完成==============", layer_index
        )

        logger.debug("===========解码器第{}层交叉注意力计算==============", layer_index)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        logger.debug(
            "===========解码器第{}层交叉注意力计算完成==============", layer_index
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        logger.debug("解码器层输出形状 -> output:{}", tgt.shape)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,  # decoder层数
        d_model: int = 256,
        return_intermediate: bool = False,  # 是否返回中间层输出
        bbox_reparam: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.return_intermediate = return_intermediate
        self.bbox_reparam = bbox_reparam

        self.layers = _get_clones(decoder_layer, num_layers)
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        self.norm = nn.LayerNorm(d_model)
        self.bbox_embed: nn.Module | None = None

    def refpoints_refine(
        self, refpoints_unsigmoid: torch.Tensor, new_refpoints_delta: torch.Tensor
    ):
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                refpoints_unsigmoid[..., :2]  # 原中心
                + new_refpoints_delta[..., :2]
                * refpoints_unsigmoid[..., 2:]  # 预测偏移 * 原宽高
            )

            new_refpoints_wh = (
                refpoints_unsigmoid[..., 2:] * new_refpoints_delta[..., 2:].exp()
            )

            new_refpoints_unsigmoid = torch.cat(
                [new_refpoints_cxcy, new_refpoints_wh], dim=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid  # b, num_queries * group_detr, 4

    def forward(
        self,
        tgt: torch.Tensor,  # b, num_queries * group_detr, d_model Embedding层权重，可学习的向量
        memory: torch.Tensor,  # b, \sum{hxw}, d_model
        # tgt_mask: torch.Tensor | None = None,
        # memory_mask: torch.Tensor | None = None,
        # tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,  # b, \sum{hxw}
        pos: torch.Tensor | None = None,  # b, \sum{hxw}, d_model 位置编码
        refpoints_unsigmoid: (
            torch.Tensor | None
        ) = None,  # b, num_queries * group_detr, 4
        # for memory
        level_start_index: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_hw: (
            list[tuple[int, int]] | None
        ) = None,  # num_levels (H, W) Python ints
        valid_ratios: torch.Tensor | None = None,  # b, 特征层层数, 2
    ):
        assert refpoints_unsigmoid is not None
        output = tgt
        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(
            refpoints: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # b, num_queries * group_detr, 4
            obj_center = refpoints[..., :4]

            # b, num_queries * group_detr, 1, 4
            a = obj_center[:, :, None]
            # b, 特征层层数, 4 ->  b, 1, 特征层层数, 4
            b = torch.cat([valid_ratios, valid_ratios], dim=-1)[:, None]

            # !多尺度：参考点映射到每个特征层
            # 广播 -> b, num_queries * group_detr, 特征层层数, 4
            refpoints_input = a * b

            # 对参考点进行位置编码 b, num_queries * group_detr, 2 * d_model
            query_sine_embed = gen_sineembed_for_position(
                refpoints_input[:, :, 0, :], self.d_model // 2
            )

            # !参考点进行位置编码，经过线性层，告诉query要关注图像哪个空间位置找
            # !tgt作为解码器的query，逐层更新，是累积出来的"内容/语义"表示回答"我这块区域里有什么东西
            #  b, num_queries * group_detr, d_model
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        for layer_index, layer in enumerate(self.layers):
            # !控制解码器每一层是否用"精炼后的参考框"来重新生成查询位置编码与可变形采样的锚点
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, _query_sine_embed = (
                    get_reference(refpoints_unsigmoid)
                )
            else:
                obj_center, refpoints_input, query_pos, _query_sine_embed = (
                    # 绝对坐标先归一化
                    get_reference(refpoints_unsigmoid.sigmoid())
                )

            # b, num_queries * group_detr, d_model
            output = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos,  # b, num_queries * group_detr, d_model
                reference_points=refpoints_input,  # b, num_queries * group_detr, 特征层层数, 4
                spatial_shapes=spatial_shapes,
                spatial_shapes_hw=spatial_shapes_hw,
                level_start_index=level_start_index,
                layer_index=layer_index,
            )

            assert self.bbox_embed is not None
            new_refpoints_delta = self.bbox_embed(
                output
            )  # 得到精修偏移量 b, num_queries * group_detr, 4

            # !精修参考框
            # b, num_queries * group_detr, 4
            new_refpoints_unsigmoid = self.refpoints_refine(
                refpoints_unsigmoid, new_refpoints_delta
            )

            # 保存精修参考框，最后一层不保存，因为最后一层没有精修
            if layer_index != self.num_layers - 1:
                hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)

            # 返回中间层的输出
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()  # 最后一层循环了添加过一次，这里剔除一下
            intermediate.append(output)

            results = []
            if self.bbox_embed:
                # num_layers, b, num_queries * group_detr, d_model
                results.append(torch.stack(intermediate))
                # num_layers, b, num_queries * group_detr, 4
                results.append(torch.stack(hs_refpoints_unsigmoid))
            else:
                results.append(torch.stack(intermediate))
                # 1, b, num_queries * group_detr, 4
                results.append(refpoints_unsigmoid.unsqueeze(0))

            logger.debug("解码器中间层输出结果：outputs: {}", results[0].shape)
            logger.debug("解码器中间层精修参考框结果：bbox: {}", results[1].shape)
            return tuple(results)

        logger.debug("解码器非中间层的最终输出：output:{}", output)
        res = output.unsqueeze(0)
        return res


def build_transformer(config: AppConfig) -> Transformer:
    return Transformer(
        d_model=config.model.d_model,
        sa_nhead=config.model.sa_nheads,
        ca_nhead=config.model.ca_nheads,
        num_queries=config.model.num_queries,
        num_decoder_layers=config.model.num_decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        activation="relu",
        dropout=config.model.dropout,
        return_intermediate_dec=True,
        group_detr=config.model.group_detr,
        two_stage=config.model.two_stage,
        num_feature_levels=len(config.model.projector_scale),
        bbox_reparam=config.model.bbox_reparam,
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
