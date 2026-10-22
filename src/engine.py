import random

import numpy as np
import torch

from src.config import load_config
from src.datasets.load_raod import build_dataloader
from tqdm import tqdm


def random_seed(seed: int = 42) -> None:
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


def train():
    args = load_config()
    random_seed(args.model.seed)
    data_loader = build_dataloader(
        True, args.train_config, patch_size=args.model.patch_size
    )

    for i, batch in tqdm(enumerate(data_loader)):
        print(batch)
        break


if __name__ == "__main__":
    train()
