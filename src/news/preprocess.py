"""
News Data Preprocessor (子任务 A)
----------------------------------
原始 CSV → news_cleaned.parquet

负责人：子任务 A
对应文件：src/news/preprocess.py
输出文件：data/processed/news/news_cleaned.parquet

运行方式：
    python -m src.news.preprocess --config configs/news_config.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path


def load_raw(finnhub_path: str, gdelt_path: str) -> pd.DataFrame:
    """
    加载并合并 finnhub 和 gdelt 原始 CSV。

    TODO (子任务 A)：
        - 读取两个 CSV，统一列名为 ticker, headline, publish_datetime, url
        - 合并为单个 DataFrame
    """
    raise NotImplementedError


def clean(df: pd.DataFrame, news_window: int = 7) -> pd.DataFrame:
    """
    清洗原始新闻，输出标准化 parquet。

    输出 schema：
        ticker (str)           — 大写，如 "AAPL"
        date (str)             — YYYY-MM-DD，该文章归属的预测日期
        article_id (str)       — 唯一标识，如 "AAPL_2024-03-15_0042"
        headline (str)         — 清洗后标题文本
        publish_datetime (datetime) — 原始发布时间（UTC）

    TODO (子任务 A)：
        1. 时区归一化：统一转为 UTC，判断是否在美东 4PM 之前
               4PM ET 之前 → 归入当日；之后 → 归入下一交易日
        2. 去重：相同 ticker + headline 的文章只保留最早一条
               可用 df.drop_duplicates(subset=["ticker", "headline"])
        3. 过滤：删除 headline 为空、长度 < 10 字符、或含乱码的行
        4. 生成 article_id：f"{ticker}_{date}_{序号:04d}"
        5. 7 天窗口映射：T 日的文章应出现在预测日期 T ~ T+6 中
               即 date 列存储的是该文章将被用于预测的日期（可能产生多行）
        6. 只保留 trading day（排除周末和美股休市日）
               可用 pandas_market_calendars 或 pandas.bdate_range 近似
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/news_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading raw news data...")
    df_raw = load_raw(config["data"]["raw_finnhub"], config["data"]["raw_gdelt"])
    print(f"  Raw rows: {len(df_raw):,}")

    print("Cleaning...")
    df_clean = clean(df_raw, news_window=config["training"]["news_window"])
    print(f"  Clean rows: {len(df_clean):,}")

    output_path = Path(config["data"]["cleaned"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Tickers: {sorted(df_clean['ticker'].unique())}")
    print(f"Date range: {df_clean['date'].min()} ~ {df_clean['date'].max()}")


if __name__ == "__main__":
    main()
