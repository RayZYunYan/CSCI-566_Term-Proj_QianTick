"""
Price Unimodal Baseline Training
----------------------------------
训练仅用价格特征的 baseline 分类器，用于消融实验。

运行方式：
    python -m src.price.train --config configs/price_config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random

from src.price.dataset import PriceDataset
from src.price.model import PriceUnimodalBaseline

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config: dict):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # TODO: 初始化 Dataset 和 DataLoader
    # train_ds = PriceDataset(split="train")
    # val_ds   = PriceDataset(split="val")

    # TODO: 初始化模型
    # model = PriceUnimodalBaseline(...).to(device)

    # TODO: 训练循环
    # - optimizer, scheduler
    # - train epoch → val epoch
    # - 保存最优 checkpoint 到 checkpoints/price_best.pt
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/price_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
