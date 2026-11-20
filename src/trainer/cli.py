"""训练、验证和预测入口的命令行参数。"""

import argparse
from pathlib import Path
from typing import Literal


type ScriptMode = Literal["train", "eval", "predict"]


def build_parser(*, mode: ScriptMode) -> argparse.ArgumentParser:
    descriptions = {
        "train": "训练 HAODDETR",
        "eval": "验证 HAODDETR 检查点",
        "predict": "使用 HAODDETR 预测本地图片",
    }
    parser = argparse.ArgumentParser(description=descriptions[mode])
    parser.add_argument(
        "--outputs", help="输出目录，默认读取 YAML，相对路径以项目根目录为基准"
    )

    if mode == "predict":
        parser.add_argument(
            "--batch-size", type=int, help="预测批次大小，默认读取 predict_config"
        )
        parser.add_argument(
            "images", metavar="IMAGE", type=Path, nargs="+", help="一张或多张图片路径"
        )
        parser.add_argument(
            "--score-threshold", type=float, help="最低置信度，默认读取 predict_config"
        )
        return parser

    parser.add_argument(
        "--dataset", choices=("raod_small", "raod"), help="RAOD 子集或完整数据集"
    )
    parser.add_argument(
        "--batch-size", type=int, help="批次大小，默认读取 train_config"
    )
    parser.add_argument("--num-workers", type=int, help="DataLoader worker 数量")
    parser.add_argument("--seed", type=int, help="随机种子")
    if mode == "train":
        parser.add_argument("--epochs", type=int, help="训练总轮数")
        parser.add_argument("--lr", type=float, help="优化器学习率")
        parser.add_argument(
            "--lr-scheduler",
            type=str.lower,
            choices=("step", "cosine"),
            help="学习率调度策略",
        )
        parser.add_argument(
            "--optimizer",
            type=str.lower,
            choices=("sgd", "adam", "adamw"),
            help="优化器类型",
        )
        parser.add_argument("--momentum", type=float, help="SGD 动量系数")
        parser.add_argument("--weight-decay", type=float, help="权重衰减系数")
        parser.add_argument(
            "--resume", action="store_true", default=None, help="从检查点恢复训练"
        )
    elif mode == "eval":
        parser.add_argument(
            "--max-detections", type=int, help="每张图参与检测指标计算的候选框数量上限"
        )

    return parser


def config_overrides(
    args: argparse.Namespace, *, mode: ScriptMode
) -> dict[str, object]:
    """仅覆盖用户显式传入的参数，保留 YAML 中的其他值。"""
    result: dict[str, object] = {
        name: value
        for name in ("outputs", "seed")
        if (value := getattr(args, name, None)) is not None
    }

    if mode == "predict":
        result["predict_config"] = {
            name: value
            for name, value in {
                "images": [str(path) for path in args.images],
                "batch_size": args.batch_size,
                "score_threshold": args.score_threshold,
            }.items()
            if value is not None
        }
        return result

    # 训练和独立验证共用 DataLoader 的数据集、批次大小和 worker 配置。
    sections = {"train_config": ("dataset", "batch_size", "num_workers")}
    if mode == "train":
        sections["train_config"] += (
            "epochs",
            "lr",
            "optimizer",
            "momentum",
            "weight_decay",
            "lr_scheduler",
            "resume",
        )
    elif mode == "eval":
        sections["evaluate_config"] = ("max_detections",)

    for section, names in sections.items():
        values = {
            name: value
            for name in names
            if (value := getattr(args, name, None)) is not None
        }
        if values:
            result[section] = values
    return result
