import random

import numpy as np
import torch
from loguru import logger

from src.config import load_config
from src.datasets.load_raod import build_dataloader
from src.utils.logger import setup_logging
from tqdm import tqdm


# def train():
#     args = load_config()
#     setup_logging(args.logger)
#     random_seed(args.model.seed)
#     logger.info(
#         "配置加载完成：实验={}，seed={}，patch_size={}",
#         args.model.name,
#         args.model.seed,
#         args.model.patch_size,
#     )
#     data_loader = build_dataloader(
#         True, args.train_config, patch_size=args.model.patch_size
#     )

#     logger.info(
#         "训练集加载完成：{} 张图片，batch_size={}，num_workers={}",
#         len(data_loader.dataset),
#         args.train_config.batch_size,
#         args.train_config.num_workers,
#     )
#     for i, batch in enumerate(tqdm(data_loader, desc="读取训练批次", unit="batch")):
#         logger.info(
#             "批次 {}：images={}，dtype={}，各图片目标数={}",
#             i,
#             tuple(batch["images"].shape),
#             batch["images"].dtype,
#             [len(boxes) for boxes in batch["boxes"]],
#         )
#         logger.debug("图片编号：{}", batch["image_ids"])
#         break


class Engine:
    def __init__(self):
        # pass
        args = load_config()
        logger.info("配置加载完成：{}", args)
        setup_logging(args.logger)
        self.random_seed(args.model.seed)
        self.args = args

    def train(self):
        train_config = self.args.train_config

        data_loader = build_dataloader(
            True, train_config, patch_size=self.args.model.patch_size
        )

    def eval():
        pass

    def random_seed(self, seed: int = 42) -> None:
        """固定全局随机源；在创建模型、DataLoader 和执行随机操作之前调用。

        NumPy 的旧式全局随机源要求种子处于 [0, 2**32) 范围。
        Albumentations Compose、np.random.default_rng 等独立生成器仍需单独设种子。
        相同种子有助于复现实验，不保证跨设备、跨版本的计算结果完全一致。
        """
        if not 0 <= seed < 2**32:
            raise ValueError("seed 必须处于 [0, 2**32) 范围")

        random.seed(seed)  # Python 随机操作，例如 random.choice。
        np.random.seed(seed)  # NumPy 全局随机操作。
        torch.manual_seed(seed)  # PyTorch CPU 及加速设备（包括 CUDA、MPS）。

        # CUDA 卷积关闭算法测速，优先使用确定性算法；这不约束其他算子。
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    engine = Engine()
    engine.train()
