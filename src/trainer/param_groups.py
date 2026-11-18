from typing import Any, cast

# import torch.nn as nn
# import torch.nn

from src.config import AppConfig
from src.models.haoddetr import HAODDETR


def build_param_dicts(config: AppConfig, model: HAODDETR) -> list[dict[str, Any]]:
    """根据模型组件和编码器层深，构建优化器的初始参数分组。

    DINOv3 已经过预训练，通常希望微调时用较小的学习率，减少对已有特征的
    扰动；检测头和投影层需要适应当前检测任务，可以使用基础学习率。
    编码器内部再逐层衰减：越靠近输入的层学习率越小，越靠近输出的层越大。
    这是一种训练策略，具体效果仍需要通过实验验证。

    此函数为每个可训练参数生成一个字典，包含参数和该组学习率；编码器
    参数还可能有单独的权重衰减。当前训练流程直接使用这些单参数组；
    本文件的 get_param_dict 可以合并超参数相同的组。Warmup 和余弦调度
    会在这些组各自的基础学习率上同步缩放。
    """
    train_config = config.train_config
    # assert isinstance(model.backbone, Joiner)
    backbone = cast("Any", model.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(
        config, prefix="backbone.0"
    )

    # 第一组：DINOv3 编码器。
    # 若编码器被冻结，requires_grad=False 的参数会被排除，此列表为空。
    # 若已解冻，get_named_param_lr_pairs 按实际层数计算逐层学习率：
    # $$ lr = lr_{encoder} \times lr_{vit\_layer\_decay}^{N+1-layer\_id} \times lr_{component\_decay}^{2} $$
    # 其中 N 是编码器总层数；越浅的层指数越大，在衰减系数小于 1 时学习率越小。
    # 该方法同时为编码器的部分参数设置独立的 weight_decay。
    backbone_param_lr_pairs = [
        param_dict for _, param_dict in backbone_named_param_lr_pairs.items()
    ]

    # 第二组：检测 Transformer 的 Decoder。仅收集需要梯度的参数。
    decoder_key = "transformer.decoder"
    decoder_params = [
        p for n, p in model.named_parameters() if decoder_key in n and p.requires_grad
    ]

    # Decoder 使用基础学习率乘组件系数；默认系数 0.7 时为基础学习率的 70%。
    # 这是组件级学习率，不是在 Decoder 内部逐层递减。
    decoder_param_lr_pairs = [
        {"params": param, "lr": train_config.lr * train_config.lr_component_decay}
        for param in decoder_params
    ]

    # 第三组：尚未归入编码器或 Decoder 的可训练参数，例如投影层、分类头和查询嵌入。
    # 根据参数名称排除前两组，避免把同一个参数再次交给优化器。
    other_params = [
        p
        for n, p in model.named_parameters()
        if (
            n not in backbone_named_param_lr_pairs
            and decoder_key not in n
            and p.requires_grad
        )
    ]

    # 这些参数使用基础学习率；weight_decay 未在字典中指定时采用优化器的默认配置。
    other_param_dicts = [
        {"params": param, "lr": train_config.lr} for param in other_params
    ]

    # 返回单参数字典列表；如需合并相同超参数的组，可通过 get_param_dict 调用。
    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
    )

    return final_param_dicts


def hyperparameter_key(param_group: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the hyperparameter overrides of ``param_group``, without its parameters.

    Values are compared by ``repr`` so a group carrying an unhashable setting (a third-party
    optimizer's list-valued option, say) still yields a key. Distinct floats keep distinct ``repr``,
    and anything without a value-based ``repr`` lands in its own bucket, which under-merges rather
    than merging two differently configured parameters.

    Args:
        param_group: A single optimizer parameter group.

    Returns:
        The group's non-``params`` items, sorted by key so groups configured identically compare equal.
    """
    # 取出params参数，使用其它参数的键值对作为key：	例如(("lr", "4e-05"), ("weight_decay", "0.0001"))
    return tuple(
        sorted(
            (key, repr(value)) for key, value in param_group.items() if key != "params"
        )
    )


def merge_buckets(param_dicts: list[dict[str, Any]]) -> list[list[int]]:
    """Bucket ``param_dicts`` indices by hyperparameter overrides, in first-appearance order.

    Args:
        param_dicts: Single-parameter groups as built by :func:`_build_param_dicts`.

    Returns:
        One list of ``param_dicts`` indices per distinct hyperparameter combination.

    # 按照相同的hyperparameter_key合并参数组
    [
        {"params": ["参数1", "参数2"], "lr": 1e-4},                       # 来自桶 [0,3]
        {"params": ["参数3"], "lr": 2e-5, "weight_decay": 1e-4},          # 来自桶 [1]
        {"params": ["参数4", "参数5"], "lr": 7e-5},                        # 来自桶 [2,5]
        {"params": ["参数6"], "lr": 4e-5, "weight_decay": 1e-4},          # 来自桶 [4]
    ]
    """
    buckets: dict[tuple[tuple[str, str], ...], list[int]] = {}
    ordered: list[list[int]] = []
    for index, param_group in enumerate(param_dicts):
        key = hyperparameter_key(param_group)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = []
            ordered.append(bucket)
        bucket.append(index)
    return ordered


def get_param_dict(args: Any, model: HAODDETR) -> list[dict[str, Any]]:
    """Build optimizer parameter groups with layer-wise LR (and backbone weight-decay) overrides.

    Parameters that end up configured identically share one group: ``torch.optim``'s foreach and
    fused kernels batch a group's parameters into a single multi-tensor launch, so one group per
    parameter would run ~500 single-tensor launches (and ~500 Python iterations of the optimizer's
    per-group loop) per step instead of one launch per distinct configuration.

    Args:
        args: Namespace supplying ``lr``, ``lr_encoder``, ``lr_component_decay``,
            ``lr_vit_layer_decay``, ``weight_decay``, and ``out_feature_indexes``.
        model: The model whose parameters the optimizer will own.

    Returns:
        Optimizer parameter groups, each holding every trainable parameter that shares its
        hyperparameter overrides.
    """
    param_dicts = build_param_dicts(args, model)
    return [
        {
            **{
                key: value
                for key, value in param_dicts[bucket[0]].items()
                if key != "params"
            },
            "params": [param_dicts[index]["params"] for index in bucket],
        }
        for bucket in merge_buckets(param_dicts)
    ]
