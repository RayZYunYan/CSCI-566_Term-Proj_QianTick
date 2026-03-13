"""
News Feature Extractor
-----------------------
用 FinBERT 处理所有新闻，生成按 (ticker, date) 索引的特征文件。

输出：data/features/news_features.parquet
    - index: MultiIndex (ticker, date)
    - columns: feat_0 ... feat_770  (771维)

运行方式：
    python -m src.news.feature_extractor
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_OUTPUT = Path("data/features/news_features.parquet")
FEAT_DIM = 771  # 768 (FinBERT CLS) + 3 (sentiment)


def extract_features() -> pd.DataFrame:
    """
    遍历所有 (ticker, date) 组合，对每个样本聚合过去7天新闻特征。

    Returns:
        df: MultiIndex (ticker, date) 的 DataFrame，771列特征

    TODO:
        - 加载 data/processed/news/ 数据
        - 初始化 FinBERTEncoder
        - 批量推理，按 (ticker, date) 聚合
        - 保存到 FEATURE_OUTPUT
    """
    raise NotImplementedError


def main():
    print("Extracting news features with FinBERT...")
    df = extract_features()
    FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_OUTPUT)
    print(f"Saved {len(df)} samples to {FEATURE_OUTPUT}")
    print(f"Shape: {df.shape}")
    print(f"Tickers: {df.index.get_level_values('ticker').unique().tolist()}")
    print(f"Date range: {df.index.get_level_values('date').min()} ~ {df.index.get_level_values('date').max()}")


if __name__ == "__main__":
    main()
