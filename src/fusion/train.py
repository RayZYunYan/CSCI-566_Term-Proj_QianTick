"""
Fusion Model Training
----------------------
训练最终的多模态融合分类器。

前置条件：
    - data/features/news_features.parquet  已生成（运行 src.news.feature_extractor）
    - data/features/price_features.parquet 已生成（运行 src.price.feature_extractor）

运行方式：
    python -m src.fusion.train --config configs/fusion_config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.fusion.dataset import FusionDataset
from src.fusion.model import QianTickModel

RANDOM_SEED = 42
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    """计算 val/test 集的 accuracy 和 AUC。TODO: 补充完整评估指标"""
    model.eval()
    # TODO: 实现评估逻辑
    raise NotImplementedError


def train(config: dict):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = FusionDataset(split="train")
    val_ds   = FusionDataset(split="val")

    # TODO: DataLoader
    # TODO: 初始化 QianTickModel
    # TODO: 训练循环，保存最优 checkpoint 到 checkpoints/fusion_best.pt
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    train(config)


if __name__ == "__main__":
    main()
