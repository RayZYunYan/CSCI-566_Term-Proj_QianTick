"""
News Unimodal Baseline Training
--------------------------------
训练仅用新闻特征的 baseline 分类器，用于消融实验。

运行方式：
    python -m src.news.train --config configs/news_config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

from src.news.dataset import NewsFeatureDataset  # 读取预提取的 771 维特征
from src.news.model import NewsUnimodalBaseline

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
    # train_ds = NewsDataset(split="train")
    # val_ds   = NewsDataset(split="val")

    # TODO: 初始化模型
    # model = NewsUnimodalBaseline(...).to(device)

    # TODO: 训练循环
    # - optimizer, scheduler
    # - train epoch → val epoch
    # - 保存最优 checkpoint 到 checkpoints/news_best.pt
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/news_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
