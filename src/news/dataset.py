"""
News Dataset
------------
NewsFeatureDataset: 加载预提取的 771 维特征 + 标签，供 train.py (子任务 D) 使用
NewsDataset:        加载原始新闻文本（保留，供未来 B/C 流水线使用）

共享时间切分约定：
    train : date <= 2024-06-30
    val   : 2024-07-01 ~ 2024-12-31
    test  : date >= 2025-01-01
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


# ── 共享约定 ───────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
           "META", "AMZN", "AMD", "INTC", "CRM", "NFLX", "AVGO"]

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"

NEWS_WINDOW = 7
RANDOM_SEED = 42

FEAT_COLS = [f"feat_{i}" for i in range(771)]  # feat_0 ~ feat_770
# ─────────────────────────────────────────────────────────


class NewsFeatureDataset(Dataset):
    """
    读取预提取的 771 维新闻特征，供 train.py (子任务 D) 使用。

    输入文件：
        features_path : news_features.parquet，含列 ticker, date, feat_0~feat_770
        labels_path   : labels.parquet，含列 ticker, date, label (0/1)

    Mock 策略（D 可立即使用，无需等 A/B/C）：

        import pandas as pd, numpy as np
        tickers = ["AAPL", "MSFT", "NVDA"]
        dates = pd.bdate_range("2023-01-01", "2025-06-01").strftime("%Y-%m-%d")

        feat_rows = [{"ticker": t, "date": d,
                      **{f"feat_{j}": np.random.randn() for j in range(771)}}
                     for t in tickers for d in dates]
        pd.DataFrame(feat_rows).to_parquet("mock_news_features.parquet")

        label_rows = [{"ticker": t, "date": d, "label": np.random.randint(0, 2)}
                      for t in tickers for d in dates]
        pd.DataFrame(label_rows).to_parquet("mock_labels.parquet")

        ds = NewsFeatureDataset("mock_news_features.parquet", "mock_labels.parquet", split="train")
        feat, label = ds[0]
        assert feat.shape == (771,)
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        split: str = "train",
    ):
        """
        Args:
            features_path: news_features.parquet 路径
            labels_path:   labels.parquet 路径
            split:         "train" | "val" | "test"
        """
        assert split in ("train", "val", "test")
        self.split = split
        self.samples = self._load(features_path, labels_path)

    def _load(self, features_path: str, labels_path: str) -> list:
        feats = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path)

        # 统一格式
        for df in (feats, labels):
            df["ticker"] = df["ticker"].str.upper()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 按 split 过滤 labels
        if self.split == "train":
            labels = labels[labels["date"] <= TRAIN_END]
        elif self.split == "val":
            labels = labels[(labels["date"] >= VAL_START) & (labels["date"] <= VAL_END)]
        else:
            labels = labels[labels["date"] >= TEST_START]

        merged = labels.merge(feats, on=["ticker", "date"], how="inner")

        samples = []
        for _, row in merged.iterrows():
            feat = torch.tensor(row[FEAT_COLS].values.astype("float32"))  # (771,)
            label = torch.tensor([int(row["label"])], dtype=torch.float32)  # (1,)
            samples.append((feat, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat:  (771,)  float32 特征向量
            label: (1,)    float32 标签（0.0 或 1.0）
        """
        return self.samples[idx]


class NewsDataset(Dataset):
    """
    每个样本对应一个 (ticker, date)，包含该日期前 NEWS_WINDOW 天的新闻文本列表
    以及当日涨跌标签（1=涨, 0=跌/平）。

    供未来 B/C 实时跑 FinBERT 的流水线使用（当前为占位实现）。

    TODO (子任务 C/B 协作):
        - 实现 _build_samples()：从 news_cleaned.parquet 读取文本
        - 实现 _load_labels()：从 labels.parquet 读取标签
        - 确认 7 天窗口对齐逻辑（避免未来数据泄露）
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")
        self.split = split
        self.samples = []
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
            "texts": texts,
            "label": torch.tensor(label, dtype=torch.long),
        }
