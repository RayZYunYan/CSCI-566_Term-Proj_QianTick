"""
Price Dataset
-------------
加载 yfinance OHLCV 数据及技术指标，供 LSTM/Transformer 训练和特征提取使用。

输入：data/raw/price/ 或 data/processed/price/
输出：(ticker, date) 索引的滑动窗口样本
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# ── 接口约定 ───────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
           "META", "AMZN", "AMD", "INTC", "CRM", "NFLX", "AVGO"]

TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"

RANDOM_SEED = 42
PRICE_WINDOW = 30  # 过去 N 天价格序列
# ─────────────────────────────────────────────────────────

# OHLCV 基础特征
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# 技术指标（TODO: 在 processed 阶段预计算好）
TECH_COLS = [
    "MA5", "MA10", "MA20",       # 移动平均
    "RSI14",                      # RSI
    "MACD", "MACD_signal",        # MACD
    "BB_upper", "BB_lower",       # 布林带
    "ATR14",                      # 平均真实波幅
]


class PriceDataset(Dataset):
    """
    每个样本对应一个 (ticker, date)，包含该日期前 PRICE_WINDOW 天的价格+指标序列
    以及当日涨跌标签（1=涨, 0=跌/平）。

    TODO:
        - 实现 _load_price() 从处理后数据读取 OHLCV + 技术指标
        - 实现 _compute_label()：next day return > 0 → 1, else → 0
        - 实现归一化/标准化（建议用训练集统计量）
        - 实现滑动窗口切分逻辑
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")
        self.split = split
        self.samples = []  # list of (ticker, date, sequence, label)
        self._build_samples()

    def _build_samples(self):
        # TODO: 实现数据加载与窗口切分
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, date, seq, label = self.samples[idx]
        return {
            "ticker": ticker,
            "date": date,
            "sequence": torch.tensor(seq, dtype=torch.float32),  # (30, feature_dim)
            "label": torch.tensor(label, dtype=torch.long),
        }
