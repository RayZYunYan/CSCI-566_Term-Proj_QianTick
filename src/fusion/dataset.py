"""
Fusion Dataset
--------------
合并新闻特征和价格特征，按 (ticker, date) 对齐，供融合模型训练使用。

输入：
    data/features/news_features.parquet   - (ticker, date) 索引，771维
    data/features/price_features.parquet  - (ticker, date) 索引，192维

输出：每个样本包含 news_feat + price_feat + label
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


NEWS_FEAT_PATH  = Path("data/features/news_features.parquet")
PRICE_FEAT_PATH = Path("data/features/price_features.parquet")

TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"

RANDOM_SEED = 42


class FusionDataset(Dataset):
    """
    按 (ticker, date) inner join 新闻特征和价格特征。
    只保留两边都有数据的样本（避免缺失）。

    TODO:
        - 确认 label 来源（建议统一从价格数据计算，避免不一致）
        - 检查 (ticker, date) 对齐后是否有大量缺失，决定是否需要插值
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")
        self.split = split
        self.data = self._load_and_merge()

    def _load_and_merge(self) -> pd.DataFrame:
        news_df  = pd.read_parquet(NEWS_FEAT_PATH)
        price_df = pd.read_parquet(PRICE_FEAT_PATH)

        # inner join on (ticker, date)
        merged = news_df.join(price_df, how="inner")

        # 按时间切分
        dates = merged.index.get_level_values("date")
        if self.split == "train":
            merged = merged[dates <= TRAIN_END]
        elif self.split == "val":
            merged = merged[(dates >= VAL_START) & (dates <= VAL_END)]
        else:
            merged = merged[dates >= TEST_START]

        print(f"[FusionDataset/{self.split}] {len(merged)} samples")
        return merged

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ticker = self.data.index[idx][0]
        date   = self.data.index[idx][1]

        # TODO: 区分 news_feat 列和 price_feat 列
        news_feat  = torch.tensor(row.filter(like="news_feat").values,  dtype=torch.float32)
        price_feat = torch.tensor(row.filter(like="price_feat").values, dtype=torch.float32)
        label      = torch.tensor(row["label"], dtype=torch.long)

        return {
            "ticker": ticker,
            "date": date,
            "news_feat": news_feat,    # (771,)
            "price_feat": price_feat,  # (192,)
            "label": label,
        }
