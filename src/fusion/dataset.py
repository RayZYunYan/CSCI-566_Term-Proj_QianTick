"""
Fusion Dataset
--------------
从 export_to_fusion/ 的 PT 文件和 news_features.parquet 加载数据，
按 (ticker, as_of_date) 对齐，供融合模型训练使用。

输入：
    export_to_fusion/features/{lstm,gru,tcn}_features.pt  - 价格侧特征
    export_to_fusion/predictions/price_model_outputs.csv  - 对齐主表（含 split 列）
    data/features/news_features.parquet                   - 新闻侧特征 (771维)

输出：每个样本包含 news_feat (771) + price_feat (320) + label_cls (0/1) + label_reg (float)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class FusionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        news_features_path: str | Path | None = None,
        price_features_dir: str | Path | None = None,
        price_csv_path: str | Path | None = None,
    ):
        assert split in ("train", "val", "test")
        self.split = split

        if news_features_path is None:
            news_features_path = REPO_ROOT / "data" / "features" / "news_features.parquet"
        if price_features_dir is None:
            price_features_dir = REPO_ROOT / "export_to_fusion" / "features"
        if price_csv_path is None:
            price_csv_path = REPO_ROOT / "export_to_fusion" / "predictions" / "price_model_outputs.csv"

        price_features_dir = Path(price_features_dir)

        # Load price feature tensors and concatenate: (N_total, 64+128+128) = (N_total, 320)
        lstm = torch.load(price_features_dir / "lstm_features.pt", weights_only=True).float()
        gru  = torch.load(price_features_dir / "gru_features.pt",  weights_only=True).float()
        tcn  = torch.load(price_features_dir / "tcn_features.pt",  weights_only=True).float()
        price_feat_all = torch.cat([lstm, gru, tcn], dim=1)  # (N_total, 320)

        # Load metadata CSV — row order matches PT files exactly.
        csv = pd.read_csv(price_csv_path)
        csv["_row_idx"] = np.arange(len(csv))

        # Load news features: (ticker, date, feat_0 ~ feat_770)
        news_df = pd.read_parquet(news_features_path)
        news_df["date"] = news_df["date"].astype(str).str[:10]
        news_min_date = news_df["date"].min()

        # Filter to requested split, then drop price rows before news coverage starts.
        csv_split = csv[csv["split"] == split].copy()
        csv_split = csv_split[csv_split["as_of_date"] >= news_min_date]

        # Join on (ticker, as_of_date == news.date).
        merged = csv_split.merge(
            news_df,
            left_on=["ticker", "as_of_date"],
            right_on=["ticker", "date"],
            how="inner",
        )

        if len(merged) == 0:
            raise ValueError(
                f"FusionDataset/{split}: no samples after joining price and news features. "
                "Check that as_of_date in price CSV matches date in news_features.parquet."
            )

        # Extract aligned price features using row indices from the full PT tensor.
        row_indices = torch.tensor(merged["_row_idx"].values, dtype=torch.long)
        self.price_feat = price_feat_all[row_indices]  # (N, 320)

        # Extract news features.
        feat_cols = [c for c in news_df.columns if c.startswith("feat_")]
        self.news_feat = torch.tensor(merged[feat_cols].values, dtype=torch.float32)  # (N, 771)

        # Labels: regression (continuous return) and classification (>0 → 1).
        returns = merged["true_label"].values.astype(np.float32)
        self.label_reg = torch.tensor(returns, dtype=torch.float32)
        self.label_cls = torch.tensor((returns > 0).astype(np.float32), dtype=torch.float32)

        print(f"[FusionDataset/{split}] {len(merged)} samples "
              f"(price rows: {len(csv_split)}, after news join: {len(merged)})")

    def __len__(self) -> int:
        return len(self.label_reg)

    def __getitem__(self, idx: int):
        return {
            "news_feat":  self.news_feat[idx],
            "price_feat": self.price_feat[idx],
            "label_cls":  self.label_cls[idx],
            "label_reg":  self.label_reg[idx],
        }
