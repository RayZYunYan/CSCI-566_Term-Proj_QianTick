"""
News Dataset
------------
加载合并后的新闻数据，供 FinBERT 特征提取和 unimodal baseline 训练使用。

输入：data/processed/news/ 下的处理后新闻
输出：(ticker, date) 索引的样本
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


# ── 接口约定 ───────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
           "META", "AMZN", "AMD", "INTC", "CRM", "NFLX", "AVGO"]

TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"

RANDOM_SEED = 42
NEWS_WINDOW = 7   # 过去 N 天新闻
# ─────────────────────────────────────────────────────────


class NewsDataset(Dataset):
    """
    每个样本对应一个 (ticker, date)，包含该日期前 NEWS_WINDOW 天的新闻文本列表
    以及当日涨跌标签（1=涨, 0=跌/平）。

    TODO:
        - 实现 _load_news() 从处理后数据读取新闻文本
        - 实现 _load_labels() 从价格数据计算涨跌标签
        - 确认新闻窗口对齐逻辑（避免未来数据泄露）
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            split: "train" | "val" | "test"
        """
        assert split in ("train", "val", "test")
        self.split = split
        self.samples = []  # list of (ticker, date, texts, label)
        self._build_samples()

    def _build_samples(self):
        # TODO: 实现数据加载逻辑
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, date, texts, label = self.samples[idx]
        return {
            "ticker": ticker,
            "date": date,
            "texts": texts,    # list of str，长度 <= NEWS_WINDOW * K
            "label": torch.tensor(label, dtype=torch.long),
        }
