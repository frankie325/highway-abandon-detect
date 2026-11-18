import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Any
from loguru import logger

from .math import MLP
from .transformer import Transformer
from .backbone.backbone import Joiner
from src.utils.tensors import NestedTensor


class HAODDETR(nn.Module):

    def __init__(
        self,
        backbone: Joiner,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        bbox_reparam: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_reparam = bbox_reparam
        self.aux_loss = aux_loss
        self.group_detr = group_detr
        self.num_queries = num_queries

        self.class_embed = nn.Linear(hidden_dim, num_classes)  # 分类预测
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # bbox预测

        query_dim = 4
        # 用来单阶段解码器参考点预测
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        # 用来单阶段解码器queries预测
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)

        self.transformer.decoder.bbox_embed = self.bbox_embed

        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )

    def forward(
        self,
        samples: NestedTensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        """forward 接收一个 NestedTensor，其包含：

            - samples.tensor: 批处理后的图像，形状为 [batch_size x 3 x H x W]
            - samples.mask: 二值掩码，形状为 [batch_size x H x W]，填充像素处为 1

        返回一个字典，包含以下元素：
            - "pred_logits": 所有查询的前景类别 logits（sigmoid 分类无独立背景类），
                            形状为 [batch_size x num_queries x num_classes]
            - "pred_boxes": 所有查询的归一化边界框坐标，表示为
                            (center_x, center_y, width, height)。这些值已归一化到 [0, 1]，
                            相对于每张图像自身尺寸（忽略可能的填充）。参见 PostProcess 以了解
                            如何还原为未归一化的边界框。
            - "aux_outputs": 可选，仅在启用辅助损失时返回。它是一个列表，
                            每个解码器层都包含上述两个键对应的字典。
        """

        features, poss, cross_attn_features = self.backbone(samples)

        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = (
                self.refpoint_embed.weight
            )  # num_queries * group_detr, 4
            query_feat_weight = (
                self.query_feat.weight
            )  # num_queries * group_detr, d_model
        else:
            # 推理阶段只取第一个group
            refpoint_embed_weight = self.refpoint_embed.weight[
                : self.num_queries
            ]  # num_queries, 4
            query_feat_weight = self.query_feat.weight[
                : self.num_queries
            ]  # num_queries, d_model

        transformer_outputs = self.transformer(
            srcs,
            masks,
            poss,
            refpoint_embed_weight,  # 单阶段参考点
            query_feat_weight,  # 单阶段queries
        )

        # 解码器层输出hs：decoder_layers, b, num_queries * group_detr, d_model
        # 解码器逐层精修框输出ref_unsigmoid：decoder_layers, b, num_queries * group_detr, 4
        # 两阶段时编码器预测的前topk个采样位置输出memory_ts：b, topk * group_detr, d_model
        # 两阶段时编码器预测的前topk个锚框boxes_ts：boxed_ts:  b, topk * group_detr, 4
        hs, ref_unsigmoid, hs_enc, ref_enc = transformer_outputs[:4]

        out: dict[str, Any] = {}
        if hs is not None:

            if self.bbox_reparam:
                # bbox_embed经过不断训练，已经学会了预测bbox的偏移量
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = (
                    outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                )
                # decoder_layers, b, num_queries * group_detr, 4
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )

            else:
                # decoder_layers, b, num_queries * group_detr, 4  非两阶段为绝对坐标，进行sigmoid归一化
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            # 预测分类 decoder_layers, b, num_queries * group_detr, num_classes
            outputs_class = self.class_embed(hs)

            # 最后一层输出
            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

            """
            中间层输出用来计算辅助损失
            [
                {   
                    pred_logits: (b, num_queries * group_detr, num_classes),
                    pred_boxes: (b, num_queries * group_detr, 4)
                }
            ...
            ]
            """
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                )

            if self.two_stage:
                assert self.transformer.enc_out_class_embed is not None
                group_detr = self.group_detr if self.training else 1
                cls_enc_list = []
                # b, topk * group_detr, d_model -> b, topk, d_model
                hs_enc_list = hs_enc.chunk(group_detr, dim=1)
                # !按照分组拆开，和相应的enc_out_class_embed配对进行预测
                for g_idx in range(group_detr):
                    cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](
                        hs_enc_list[g_idx]
                    )
                    # b, topk, n_classes
                    cls_enc_list.append(cls_enc_gidx)
                # b, topk * group_detr, n_classes
                cls_enc = torch.cat(cls_enc_list, dim=1)
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        for i, output in enumerate(out.get("aux_outputs", [])):
            logger.debug(
                "Decoder中间层{}用于辅助损失的预测输出pred_logits：{}, pred_boxes{}",
                i + 1,
                output["pred_logits"].shape,
                output["pred_boxes"].shape,
            )

        logger.debug(
            "最后一层Decoder层输出分类out.pred_logits: {}", out["pred_logits"].shape
        )
        if self.two_stage:
            logger.debug(
                "两阶段时编码器输出分类out.enc_outputs.pred_logits: {}",
                out["enc_outputs"]["pred_logits"].shape,
            )
            logger.debug(
                "两阶段时编码器输出精修框out.enc_outputs.pred_boxes: {}",
                out["enc_outputs"]["pred_boxes"].shape,
            )

        """
        out = {
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
        """
        return out

    def _set_aux_loss(self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor):

        # 这里去掉最后一个层的输出，把最终层留给主损失，只对中间层做辅助监督
        # decoder_layers - 1 , b, num_queries * group_detr, num_classes
        # decoder_layers - 1 , b, num_queries * group_detr, 4
        values: list[Any] = [outputs_class[:-1], outputs_coord[:-1]]
        aux_outputs = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(*values)]
        # logger.debug("用来计算辅助损失的中间层输出 aux_outputs:{}", aux_outputs)
        return aux_outputs
