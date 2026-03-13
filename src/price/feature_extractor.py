"""
Price Feature Extractor
------------------------
从 OHLCV 数据提取价格特征，生成按 (ticker, date) 索引的特征文件。

输出：data/features/price_features.parquet
    - index: MultiIndex (ticker, date)
    - columns: price_feat_0 ... price_feat_191  (128 + 64 = 192维)

运行方式：
    python -m src.price.feature_extractor
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

RAW_PRICE_DIR    = Path("data/raw/price")
PROCESSED_PRICE_DIR = Path("data/processed/price")
FEATURE_OUTPUT   = Path("data/features/price_features.parquet")

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
           "META", "AMZN", "AMD", "INTC", "CRM", "NFLX", "AVGO"]
START_DATE = "2022-01-01"
END_DATE   = "2025-12-31"


def download_price_data():
    """
    用 yfinance 下载所有 ticker 的 OHLCV 数据，保存到 data/raw/price/。

    TODO: 增量下载逻辑（已有数据不重复下载）
    """
    RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        out_path = RAW_PRICE_DIR / f"{ticker}.csv"
        if out_path.exists():
            print(f"{ticker}: already downloaded, skipping.")
            continue
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        df.to_csv(out_path)
        print(f"{ticker}: downloaded {len(df)} rows → {out_path}")


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 OHLCV DataFrame 上计算技术指标，返回添加指标列后的 DataFrame。

    TODO: 实现各指标计算（建议用 pandas-ta 或 ta-lib）
    """
    # 示例：MA
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    # TODO: RSI, MACD, Bollinger Bands, ATR
    return df


def extract_features() -> pd.DataFrame:
    """
    加载价格数据，计算指标，用预训练的 LSTM 提取特征。

    Returns:
        df: MultiIndex (ticker, date) 的 DataFrame，192列特征

    TODO:
        - 加载训练好的 PriceUnimodalBaseline checkpoint
        - 推理提取中间层特征（不包含 classifier head）
        - 按 (ticker, date) 整理输出
    """
    raise NotImplementedError


def main():
    print("Downloading price data...")
    download_price_data()

    print("Extracting price features...")
    df = extract_features()
    FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_OUTPUT)
    print(f"Saved {len(df)} samples to {FEATURE_OUTPUT}")


if __name__ == "__main__":
    main()
