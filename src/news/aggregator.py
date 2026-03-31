"""
News Feature Aggregator + Feature Extraction Pipeline (子任务 C)
----------------------------------------------------------------
输入：news_article_embeddings.parquet（子任务 B 的输出），每行 771 维
输出：news_features.parquet，MultiIndex (ticker, date)，列 feat_0~feat_770

负责人：子任务 C
对应文件：src/news/aggregator.py, src/news/feature_extractor.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path


class NewsFeatureAggregator(nn.Module):
    """
    将同一 (ticker, date) 窗口内多篇文章的 771 维向量聚合为单个 771 维向量。

    TODO (子任务 C)：
        1. mean 模式：article_embeddings.mean(dim=0)
        2. attention 模式：
               score = softmax(self.attn_w @ article_embeddings.T)  # (N,)
               output = (score.unsqueeze(-1) * article_embeddings).sum(dim=0)
        3. 边界情况：
               N == 0 → 返回 torch.zeros(feature_dim)
               N == 1 → 直接返回 article_embeddings[0]
               N > 50 → 只保留最新 50 篇（调用方按时间排序后传入）

    Mock 接口（供 D 开发时验证输出形状）：
        agg = NewsFeatureAggregator(mode="mean")
        embeddings = torch.randn(7, 771)   # 7 篇文章
        feature = agg(embeddings)          # 期望 shape: (771,)
    """

    def __init__(self, mode: str = "mean", feature_dim: int = 771):
        """
        Args:
            mode: "mean" | "attention"
            feature_dim: 输入/输出特征维度，默认 771
        """
        super().__init__()
        assert mode in ("mean", "attention"), f"Unknown mode: {mode}"
        self.mode = mode
        self.feature_dim = feature_dim
        if mode == "attention":
            # TODO: 定义可学习的 attention 权重向量
            self.attn_w = nn.Parameter(torch.randn(feature_dim))

    def forward(self, article_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            article_embeddings: (N, 771)，N 为窗口内文章数；N=0 时返回全零

        Returns:
            feature: (771,)
        """
        # TODO: 实现聚合逻辑
        raise NotImplementedError


def extract_features(config: dict) -> pd.DataFrame:
    """
    读取 news_article_embeddings.parquet（B 的输出），按 7 天窗口聚合，
    输出覆盖所有 (ticker, trading_date) 的特征 DataFrame。

    Args:
        config: 从 news_config.yaml 加载的配置字典

    Returns:
        df: MultiIndex (ticker, date)，列 feat_0~feat_770

    TODO (子任务 C)：
        - 读取 config["data"]["article_embeddings"]
        - 读取 price 模块的日期列表，确定需要覆盖的所有 (ticker, date)
        - 对每个 (ticker, date)，取过去 news_window 天内的文章（按时间倒序取最多 50 篇）
        - 调用 NewsFeatureAggregator.forward() 聚合
        - 缺失 (ticker, date) 用全零向量填充，不能漏掉（否则 Fusion 会报错）
        - 保存到 config["data"]["news_features"]
    """
    raise NotImplementedError


def main(config_path: str = "configs/news_config.yaml"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = config["data"]["news_features"]
    print("Aggregating article embeddings into daily features...")
    df = extract_features(config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved {len(df)} (ticker, date) samples to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Tickers: {df.index.get_level_values('ticker').unique().tolist()}")
    print(f"Date range: {df.index.get_level_values('date').min()} ~ "
          f"{df.index.get_level_values('date').max()}")


if __name__ == "__main__":
    main()
