from PIL.Image import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import math

from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"_is_power_of_2：无效的输入 : {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


# 多尺度多头注意力
class MSDeformAttn(nn.Module):
    def __init__(
        self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model必须能被n_heads整除, 但是得到的是d_model： {d_model} 和n_heads: {n_heads}"
            )

        d_k = d_model // n_heads
        if not _is_power_of_2(d_k):
            logger.warn(
                "最好设置每个头的维度为2的幂次，有利于cuda加速计算，但是得到的是d_k： {d_k}",
                d_k,
            )
        self.d_model = d_model
        self.n_levels = n_levels  # 多尺度
        self.n_heads = n_heads  # 多头
        self.n_points = n_points  # 采样点数
        self.d_k = d_model // n_heads  # 每个头的维度

        # value预测
        self.value_proj = nn.Linear(d_model, d_model)
        # 采样点预测
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 注意力权重预测
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.output_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self) -> None:
        # === sampling_offsets 初始化：用「规则网格」作为初始采样偏移 ===
        # 思路：先把 weight 置 0，使初始偏移完全由设计的 bias 决定；
        # 再为每个 head 分配一个方向、为每个采样点分配递增半径，形成
        # "参考点周围、不同方向、不同距离" 的初始采样网格，提供强空间先验、加速收敛。
        constant_(
            self.sampling_offsets.weight.data, 0.0
        )  # weight 置 0 → 初始偏移完全由 bias 决定
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )  # 为每个 head 分配均匀分布在圆周上的角度 n_head, 即 0, 2π/n_head, 4π/n_head, ..., (n_head-1)2π/n_head
        grid_init = torch.stack(
            [thetas.cos(), thetas.sin()], -1
        )  # 角度 → 单位方向向量 (cos, sin)，每头一个方向, n_head, 2 ：[[cos0, sin0], [cos1, sin1], ..., [cos(n_head-1), sin(n_head-1)]
        grid_init = (
            (
                grid_init / grid_init.abs().max(-1, keepdim=True)[0]
            )  # 归一化到最大分量约 ±1，保持方向不变
            .view(self.n_heads, 1, 1, 2)
            .repeat(
                1, self.n_levels, self.n_points, 1
            )  # 扩展到 n_heads, n_levels, n_points, 2 各层各采样点初始方向相同
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (
                i + 1
            )  # 同一头内第 i 个采样点距离按 i+1 递增 → 由近及远的射线
        with torch.no_grad():
            # 将规则网格展平设为 bias；no_grad 仅表示构建时不追踪梯度，该 Parameter 后续仍可被训练更新
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,  # b, num_queries * group_detr, d_model
        reference_points: torch.Tensor,  # b, num_queries * group_detr, 特征层层数, 4
        input_flatten: torch.Tensor,  # b, \sum{hxw}, d_model
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor,  # b, \sum{hxw}
        input_spatial_shapes_hw: list[tuple[int, int]],
    ) -> torch.Tensor:

        batch_size, len_input, _ = input_flatten.shape
        batch_size, len_query, _ = query.shape

        # 检查多尺度特征图所有空间位置的总数，是否等于 input_flatten 中的特征数量
        assert (
            input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]
        ).sum() == len_input

        # 由memory叠加线性层预测value
        value = self.value_proj(input_flatten)  # b, \sum{hxw}, d_model
        if input_padding_mask is not None:
            # 添加mask
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # 线性层预测权重 b, num_queries * group_detr, n_heads, n_levels * n_points
        attention_weights = self.attention_weights(query).view(
            batch_size, len_query, self.n_heads, self.n_levels * self.n_points
        )

        # b, num_queries * group_detr, n_heads * n_levels * n_points * 2
        # ->  b, num_queries * group_detr, n_heads, n_levels, n_points, 2
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )

        # 这是参考点为x,y的情况，代码不走这
        if reference_points.shape[-1] == 2:
            pass
        elif reference_points.shape[-1] == 4:
            # !参考点为四个坐标 cx, cy, w, h，已归一化

            # reference_points.shape =  b, num_queries * group_detr, 特征层层数, 4
            #  b, num_queries * group_detr, n_heads, 特征层层数, n_points, 4 -> 广播
            #  b, num_queries * group_detr, n_heads, 特征层层数, n_points, 2

            # 采样点 = 参考点xy + 预测偏移 * 参考点wh * 0.5
            # ?sampling_offsets为什么要除以n_points
            # sampling_offsets一开始的输出 = bias = grid_init, 抵消初始化时采样点的 i+1 倍放大，重新归一化回合理范围
            # ?为什么要乘以0.5
            # 把框的全宽全高 (w, h) 变成"半宽半高" (w/2, h/2)，
            # 让采样偏移以框中心为原点、覆盖的范围正好是框本身，而不是框的 2 倍大
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )  # b, num_queries * group_detr, n_heads, 特征层层数, n_points, 2
        else:
            raise ValueError(
                "reference_points 的最后一维必须是 2（中心点 cxcy）或 4（边界框 cxcywh），"
                f"当前为 {reference_points.shape[-1]}"
            )

        # b, n_heads, d_k, \sum{hxw}
        value = (
            value.transpose(1, 2)  # b, d_model, \sum{hxw}
            .contiguous()
            .view(batch_size, self.n_heads, self.d_k, len_input)
        )

        # !同一个 query、同一个 head 的所有特征层的采样权重之和为 1
        # b, num_queries * group_detr, n_heads, n_levels * n_points
        attention_weights = F.softmax(attention_weights, dim=-1)

        # b, num_queries * group_detr, d_model
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=input_spatial_shapes_hw,
        )

        # 最后再经过线性层  b, num_queries * group_detr, d_model
        return self.output_proj(output)


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_spatial_shapes_hw: list[tuple[int, int]] | None = None,
):
    batch_size, n_heads, d_k, len_input = value.shape  # b, n_heads, d_k, \sum{hxw}
    _, np, n_heads, n_levels, n_points, _ = sampling_locations.shape

    shapes = (
        value_spatial_shapes_hw
        if value_spatial_shapes_hw is not None
        else value_spatial_shapes
    )
    # num_levels = len(shapes)  # 特征层层数

    # 按特征层拆分 [ (b, n_heads, d_k, h*w,), ... ]
    value_list = value.split(
        [int(height) * int(width) for height, width in shapes], dim=3
    )

    # 将预测坐标转换到 grid_sample 的范围 [-1, 1]
    # b, num_queries * group_detr, n_heads, 特征层层数, n_points, 2
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    # 循环处理每个特征层
    for level_index, (height, width) in enumerate(shapes):

        # b*n_heads, d_k, h, w
        value_level = value_list[level_index].view(
            batch_size * n_heads, d_k, height, width
        )

        # b, num_queries * group_detr, n_heads, n_points, 2
        # -> b, n_heads, num_queries * group_detr, n_points, 2
        # ->  b*n_heads, num_queries * group_detr, n_points, 2
        sampling_grid_level = (
            sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        )

        # !双线性插值采样，从value_level特征图中得到每个预测点位对应的特征值
        #  b*n_heads, d_k, num_queries * group_detr, n_points
        sampling_value_level = F.grid_sample(
            value_level,  # b*n_heads, d_k, h, w
            sampling_grid_level,  # 最后一维必须要是x, y, sampling_grid_level的同一组坐标会分别作用于所有 d_k 个通道
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        sampling_value_list.append(sampling_value_level)

    # b, num_queries * group_detr, n_heads, n_levels * n_points
    # -> b, n_heads, num_queries * group_detr, n_levels * n_points
    # ->  b*n_heads, 1, num_queries * group_detr, n_levels * n_points  1会广播到d_k
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, np, n_levels * n_points
    )
    # b*n_heads, d_k, num_queries * group_detr, n_levels, n_points
    # -> b*n_heads, d_k, num_queries * group_detr, n_levels*n_points
    sampling_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # logger.debug("sampling_values.shape:{}", sampling_values.shape)
    # logger.debug("attention_weights.shape:{}", attention_weights.shape)
    # !特征层的所有点位进行加权求和
    # b*n_heads, d_k, num_queries * group_detr
    scores = (sampling_values * attention_weights).sum(-1)
    # b, n_heads*d_k, num_queries * group_detr
    output = scores.view(batch_size, n_heads * d_k, np)

    # b, num_queries * group_detr, d_model
    return output.transpose(1, 2).contiguous()
