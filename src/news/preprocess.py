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
import hashlib
import re
from datetime import time
from typing import Optional
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

_ET = ZoneInfo("America/New_York")


class _NYSECalendar(AbstractHolidayCalendar):
    """Approximate modern NYSE holiday calendar for this project."""

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date="2022-06-19",
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def _build_trading_days(start: str, end: str) -> "list[str]":
    """
    Return sorted list of NYSE trading day strings (YYYY-MM-DD).
    Uses an approximate NYSE holiday calendar:
    New Year, MLK, Presidents, Good Friday, Memorial, Juneteenth (2022+),
    Independence Day, Labor Day, Thanksgiving, Christmas.
    """
    bdays = pd.bdate_range(start, end, freq=CustomBusinessDay(calendar=_NYSECalendar()))
    return list(bdays.strftime("%Y-%m-%d"))


# ── Deduplication helpers ──────────────────────────────────────────────────────

def _norm_headline(text: str) -> str:
    """Normalize headline for similarity-based dedup: lowercase, strip punct."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]                           # cap length to collapse near-dupes


def _clean_headline(text: str) -> str:
    """Strip and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _is_garbled(text: str) -> bool:
    """True if >30 % of characters are non-ASCII printable."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) > 0.30


# ── Load ───────────────────────────────────────────────────────────────────────

def load_raw(finnhub_path: str, gdelt_path: str) -> pd.DataFrame:
    """
    Load and merge finnhub (per-ticker) and gdelt (market-wide) CSVs.
    Returns a DataFrame with columns:
        ticker, publish_datetime (UTC-aware), headline, url
    GDELT articles have no ticker in the source; they are broadcast to all
    tickers found in finnhub since they carry general tech-sector signals.
    """
    # ── Finnhub ────────────────────────────────────────────────────────────────
    f = pd.read_csv(finnhub_path, usecols=["symbol", "datetime", "headline", "url"])
    f.rename(columns={"symbol": "ticker", "datetime": "publish_datetime"}, inplace=True)
    f["ticker"] = f["ticker"].str.upper().str.strip()
    # Finnhub datetimes are naive UTC (verified via datetime_unix)
    f["publish_datetime"] = pd.to_datetime(f["publish_datetime"], utc=True)

    # ── GDELT ─────────────────────────────────────────────────────────────────
    g = pd.read_csv(gdelt_path, usecols=["datetime", "headline", "url"])
    g.rename(columns={"datetime": "publish_datetime"}, inplace=True)
    g["publish_datetime"] = pd.to_datetime(g["publish_datetime"], utc=True)

    # Deduplicate GDELT by normalized headline BEFORE fan-out to tickers
    g["_norm"] = g["headline"].apply(_norm_headline)
    g.drop_duplicates(subset=["_norm"], inplace=True)
    g.drop(columns=["_norm"], inplace=True)
    g.reset_index(drop=True, inplace=True)

    # Broadcast to all tickers (GDELT = market-wide tech news)
    all_tickers = sorted(f["ticker"].unique())
    g = pd.concat([g.assign(ticker=t) for t in all_tickers], ignore_index=True)

    df = pd.concat([f, g], ignore_index=True)
    return df


# ── Clean ──────────────────────────────────────────────────────────────────────

def clean(
    df: pd.DataFrame,
    news_window: int = 7,
    min_headline_len: int = 10,
    garbled_ratio_threshold: float = 0.30,
    cutoff_hour_et: int = 16,
) -> pd.DataFrame:
    """
    Clean raw news and produce the standard cleaned schema.

    Output schema (all rules enforced):
        ticker          str        — uppercase, e.g. "AAPL"
        date            str        — YYYY-MM-DD prediction date
        article_id      str        — unique article identifier
        headline        str        — cleaned headline text
        publish_datetime datetime  — original publish time (UTC-aware)

    Rules applied:
        1. Headline filter: remove null / len<10 / garbled
        2. Similarity dedup: (ticker, normalized_headline) hash — keep earliest
        3. Timezone cutoff: after 4 PM ET → next NYSE trading day
        4. 7-day window: each article maps to prediction dates [T, T+6] trading days
        5. trading-days-only: weekends + NYSE holidays excluded
    """
    df = df.copy()

    # 1. Clean & filter headlines
    df["headline"] = df["headline"].apply(_clean_headline)
    mask = (
        df["headline"].notna()
        & (df["headline"].str.len() >= min_headline_len)
    )
    df = df[mask].copy()
    non_ascii_ratio = df["headline"].apply(lambda text: sum(1 for c in text if ord(c) > 127) / len(text))
    df = df[non_ascii_ratio <= garbled_ratio_threshold].copy()

    # 2. Similarity dedup: normalize headline → hash → (ticker, hash) key
    #    Keeps the earliest article (lowest publish_datetime) per normalized pair
    df["_norm_hl"] = df["headline"].apply(_norm_headline)
    df.sort_values("publish_datetime", inplace=True)
    df.drop_duplicates(subset=["ticker", "_norm_hl"], keep="first", inplace=True)
    df.drop(columns=["_norm_hl"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Build trading-day calendar for the full data range
    min_date = df["publish_datetime"].min().strftime("%Y-%m-%d")
    max_date = (df["publish_datetime"].max() + pd.Timedelta(days=news_window + 7)).strftime("%Y-%m-%d")
    trading_days = _build_trading_days(min_date, max_date)
    trading_day_set = set(trading_days)
    td_arr = np.array(trading_days)
    td_index = {d: i for i, d in enumerate(trading_days)}

    # 4. Determine prediction base date per article
    et_times = df["publish_datetime"].dt.tz_convert(_ET)
    publish_date_et = et_times.dt.strftime("%Y-%m-%d")

    # Strictly after 4 PM ET -> next trading day.
    # Exactly 4:00 PM ET stays on the same day.
    is_after_close = et_times.dt.time > time(hour=cutoff_hour_et, minute=0)

    def _pred_base(pub_date: str, after_close: bool) -> Optional[int]:
        """Return td_arr index of the first prediction date, or None if out of range."""
        if not after_close and pub_date in td_index:
            return td_index[pub_date]
        # need to advance by at least 1 trading day
        start_dt = pd.Timestamp(pub_date) + pd.Timedelta(days=1)
        for _ in range(14):
            candidate = start_dt.strftime("%Y-%m-%d")
            if candidate in td_index:
                return td_index[candidate]
            start_dt += pd.Timedelta(days=1)
        return None

    df["_base_idx"] = [
        _pred_base(d, ac) for d, ac in zip(publish_date_et, is_after_close)
    ]
    df.dropna(subset=["_base_idx"], inplace=True)
    df["_base_idx"] = df["_base_idx"].astype(int)

    # 5. Generate article_id BEFORE window expansion (stable across prediction dates)
    #    Format: {ticker}_{publish_date_ET}_{first8_of_md5(ticker|norm_headline)}
    def _make_article_id(row) -> str:
        raw = f"{row['ticker']}|{_norm_headline(row['headline'])}"
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{row['ticker']}_{publish_date_et[row.name]}_{h}"

    df["article_id"] = df.apply(_make_article_id, axis=1)

    # 6. Vectorized 7-day window expansion via cross-join with offset table
    offsets_df = pd.DataFrame({"_offset": range(news_window), "_key": 1})
    core = df[["ticker", "_base_idx", "article_id", "headline", "publish_datetime"]].copy()
    core["_key"] = 1

    expanded = core.merge(offsets_df, on="_key").drop(columns=["_key"])
    expanded["_date_idx"] = expanded["_base_idx"] + expanded["_offset"]
    expanded = expanded[expanded["_date_idx"] < len(td_arr)].copy()
    expanded["date"] = td_arr[expanded["_date_idx"].values]

    # 7. Final schema
    result = expanded[["ticker", "date", "article_id", "headline", "publish_datetime"]].copy()
    result["ticker"] = result["ticker"].str.upper()
    result["date"] = result["date"].astype(str)
    result["publish_datetime"] = pd.to_datetime(result["publish_datetime"], utc=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

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
    preprocess_cfg = config.get("preprocess", {})
    df_clean = clean(
        df_raw,
        news_window=config["training"]["news_window"],
        min_headline_len=preprocess_cfg.get("min_headline_len", 10),
        garbled_ratio_threshold=preprocess_cfg.get("garbled_non_ascii_ratio", 0.30),
        cutoff_hour_et=preprocess_cfg.get("cutoff_hour_et", 16),
    )
    print(f"  Clean rows: {len(df_clean):,}")

    output_path = Path(config["data"]["cleaned"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Tickers:    {sorted(df_clean['ticker'].unique())}")
    print(f"Date range: {df_clean['date'].min()} ~ {df_clean['date'].max()}")


if __name__ == "__main__":
    main()
