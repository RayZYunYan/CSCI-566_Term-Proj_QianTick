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

from src.news.preprocess import _build_trading_days


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
            self.attn_w = nn.Parameter(torch.randn(feature_dim))

    def forward(self, article_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            article_embeddings: (N, 771)，N 为窗口内文章数；N=0 时返回全零

        Returns:
            feature: (771,)
        """
        if article_embeddings.ndim != 2:
            raise ValueError(
                f"article_embeddings must be 2D (N, D), got shape {tuple(article_embeddings.shape)}"
            )
        if article_embeddings.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: expected {self.feature_dim}, got {article_embeddings.shape[1]}"
            )

        n = article_embeddings.shape[0]
        if n == 0:
            return torch.zeros(
                self.feature_dim,
                dtype=article_embeddings.dtype,
                device=article_embeddings.device,
            )
        if n == 1:
            return article_embeddings[0]
        if n > 50:
            article_embeddings = article_embeddings[-50:]

        if self.mode == "mean":
            return article_embeddings.mean(dim=0)

        scores = torch.softmax(article_embeddings @ self.attn_w, dim=0)  # (N,)
        return (scores.unsqueeze(-1) * article_embeddings).sum(dim=0)


def _load_price_coverage_pairs(config: dict, article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mandatory (ticker, date) coverage list.
    Priority:
      1) labels.parquet if available
      2) price_features.parquet if available
      3) fallback to article embedding date span x configured tickers
    """
    data_cfg = config.get("data", {})
    labels_path = Path(data_cfg.get("labels", ""))
    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        if {"ticker", "date"}.issubset(labels_df.columns):
            pairs = labels_df[["ticker", "date"]].copy()
            pairs["ticker"] = pairs["ticker"].astype(str).str.upper()
            pairs["date"] = pd.to_datetime(pairs["date"]).dt.strftime("%Y-%m-%d")
            return pairs.drop_duplicates().reset_index(drop=True)

    price_feat_path = Path("data/features/price_features.parquet")
    if price_feat_path.exists():
        pf = pd.read_parquet(price_feat_path)
        if {"ticker", "date"}.issubset(pf.columns):
            pairs = pf[["ticker", "date"]].copy()
        elif list(pf.index.names) == ["ticker", "date"]:
            pairs = pf.index.to_frame(index=False)[["ticker", "date"]]
        else:
            pairs = pd.DataFrame()
        if not pairs.empty:
            pairs["ticker"] = pairs["ticker"].astype(str).str.upper()
            pairs["date"] = pd.to_datetime(pairs["date"]).dt.strftime("%Y-%m-%d")
            return pairs.drop_duplicates().reset_index(drop=True)

    tickers = config.get("tickers", [])
    if not tickers:
        tickers = sorted(article_df["ticker"].astype(str).str.upper().unique().tolist())
    if article_df.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    start = article_df["date"].min()
    end = article_df["date"].max()
    trading_days = _build_trading_days(start, end)
    fallback = pd.MultiIndex.from_product(
        [sorted({str(t).upper() for t in tickers}), trading_days],
        names=["ticker", "date"],
    ).to_frame(index=False)
    return fallback


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
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    article_path = Path(data_cfg["article_embeddings"])
    if not article_path.exists():
        raise FileNotFoundError(f"Article embedding file not found: {article_path}")

    df = pd.read_parquet(article_path)
    expected_meta_cols = {"ticker", "date", "article_id"}
    feat_cols = [f"feat_{i}" for i in range(model_cfg.get("feature_dim", 771))]
    missing = (expected_meta_cols | set(feat_cols)) - set(df.columns)
    if missing:
        raise ValueError(f"article_embeddings missing columns: {sorted(missing)}")

    df = df[["ticker", "date", "article_id"] + feat_cols].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cover_pairs = _load_price_coverage_pairs(config, df)
    if cover_pairs.empty:
        return pd.DataFrame(columns=["ticker", "date"] + feat_cols)
    cover_pairs["ticker"] = cover_pairs["ticker"].astype(str).str.upper()
    cover_pairs["date"] = pd.to_datetime(cover_pairs["date"]).dt.strftime("%Y-%m-%d")
    cover_pairs = cover_pairs.drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)

    article_grouped = {}
    for (ticker, date), g in df.groupby(["ticker", "date"], sort=False):
        emb = torch.tensor(g[feat_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        article_grouped[(ticker, date)] = emb

    agg = NewsFeatureAggregator(
        mode=model_cfg.get("aggregation", "mean"),
        feature_dim=model_cfg.get("feature_dim", 771),
    )
    agg.eval()

    rows = []
    for row in cover_pairs.itertuples(index=False):
        key = (row.ticker, row.date)
        emb = article_grouped.get(key, torch.zeros(0, len(feat_cols), dtype=torch.float32))
        with torch.no_grad():
            feat = agg(emb).cpu().numpy().astype(np.float32)
        out = {"ticker": row.ticker, "date": row.date}
        out.update({feat_cols[i]: float(feat[i]) for i in range(len(feat_cols))})
        rows.append(out)

    out_df = pd.DataFrame(rows, columns=["ticker", "date"] + feat_cols)
    out_df.sort_values(["ticker", "date"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)

    return out_df


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
