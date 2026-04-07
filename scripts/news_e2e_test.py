"""
News Module End-to-End Integration Test (子任务 A)
--------------------------------------------------
验证 A → B → C → D 四个子任务能串联跑通。

Step A: 始终使用真实 CSV，调用 src/news/preprocess.py 的实际流水线。
Step B/C/D: 若真实产物不存在，则用 mock 数据代替（兜底，方便并行开发）。

运行方式：
    python scripts/news_e2e_test.py           # Step A 用真实数据；B/C/D 真实或 mock
    python scripts/news_e2e_test.py --mock-bcd  # Step A 真实；B/C/D 强制 mock
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "news_config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

FEAT_COLS = [f"feat_{i}" for i in range(771)]
TICKERS   = ["AAPL", "MSFT", "NVDA"]
DATES     = pd.bdate_range("2023-01-01", "2025-06-01").strftime("%Y-%m-%d").tolist()


# ── Mock data generators ───────────────────────────────────────────────────────

def make_mock_article_embeddings(path: Path):
    """Mock news_article_embeddings.parquet (subtask B output)."""
    rows = []
    for t in TICKERS:
        for i, d in enumerate(DATES[:50]):
            row = {"ticker": t, "date": d, "article_id": f"{t}_{d}_{i:04d}"}
            row.update({c: float(np.random.randn()) for c in FEAT_COLS})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  [mock] Created {path} ({len(rows)} rows)")


def make_mock_news_features(path: Path):
    """
    Mock news_features.parquet (subtask C output).
    Stored as flat columns (ticker, date, feat_0..feat_770) so that
    NewsFeatureDataset (dataset.py) can read it without a reset_index call.
    NOTE: when aggregator.py is implemented it should also save flat columns,
    or dataset.py should be updated to call reset_index after reading.
    """
    rows = []
    for t in TICKERS:
        for d in DATES[:200]:
            row = {"ticker": t, "date": d}
            row.update({c: float(np.random.randn()) for c in FEAT_COLS})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  [mock] Created {path} ({len(rows)} rows)")


def make_mock_labels(path: Path):
    """Mock labels.parquet."""
    rows = [{"ticker": t, "date": d, "label": int(np.random.randint(0, 2))}
            for t in TICKERS for d in DATES[:200]]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  [mock] Created {path} ({len(rows)} rows)")


# ── Step checkers ──────────────────────────────────────────────────────────────

def check_step_a():
    """
    Step A: always run the real preprocessing pipeline.
    Validates schema, trading-day constraint, 7-day window, and article_id uniqueness.
    """
    print("\n[Step A] Running real preprocessing pipeline...")
    from src.news.preprocess import _build_trading_days, load_raw, clean

    cfg_data = CONFIG["data"]
    cfg_train = CONFIG["training"]
    cfg_pre = CONFIG.get("preprocess", {})

    finnhub_path = ROOT / cfg_data["raw_finnhub"]
    gdelt_path   = ROOT / cfg_data["raw_gdelt"]
    out_path     = ROOT / cfg_data["cleaned"]

    assert finnhub_path.exists(), f"Missing finnhub CSV: {finnhub_path}"
    assert gdelt_path.exists(),   f"Missing GDELT CSV:   {gdelt_path}"

    df_raw = load_raw(str(finnhub_path), str(gdelt_path))
    assert len(df_raw) > 0, "load_raw returned empty DataFrame"

    df = clean(
        df_raw,
        news_window=cfg_train["news_window"],
        min_headline_len=cfg_pre.get("min_headline_len", 10),
        garbled_ratio_threshold=cfg_pre.get("garbled_non_ascii_ratio", 0.30),
        cutoff_hour_et=cfg_pre.get("cutoff_hour_et", 16),
    )

    # ── Schema checks ──────────────────────────────────────────────────────────
    required_cols = {"ticker", "date", "article_id", "headline", "publish_datetime"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert df["ticker"].str.isupper().all(), "ticker contains lowercase"
    assert df["headline"].notna().all(), "headline has null values"
    assert (df["headline"].str.len() >= cfg_pre.get("min_headline_len", 10)).all(), \
        "headline shorter than configured threshold"
    assert str(df["publish_datetime"].dtype) == "datetime64[ns, UTC]", \
        f"publish_datetime dtype mismatch: {df['publish_datetime'].dtype}"

    # ── Trading-day check ──────────────────────────────────────────────────────
    sample_dates = df["date"].drop_duplicates().sample(min(500, df["date"].nunique()), random_state=42)
    valid_trading_days = set(_build_trading_days(df["date"].min(), df["date"].max()))
    for d in sample_dates:
        assert d in valid_trading_days, f"date {d} is not a valid trading day"

    # Sanity check for known NYSE days:
    # - Good Friday closed
    # - Columbus Day / Veterans Day open
    known_days = set(_build_trading_days("2025-04-01", "2025-11-30"))
    assert "2025-04-18" not in known_days, "Good Friday should not be a trading day"
    assert "2025-10-13" in known_days, "Columbus Day should remain a trading day"
    assert "2025-11-11" in known_days, "Veterans Day should remain a trading day"

    # ── 7-day window: publish_datetime must be ≤ date ─────────────────────────
    pub_date = df["publish_datetime"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    assert (pub_date <= df["date"]).all(), \
        "Some articles have prediction date BEFORE publish date"

    # ── article_id format sanity ───────────────────────────────────────────────
    assert df["article_id"].str.match(r"^[A-Z]+_\d{4}-\d{2}-\d{2}_[0-9a-f]+$").all(), \
        "article_id format mismatch"

    # ── Uniqueness: (ticker, date, article_id) should be unique ───────────────
    dupes = df.duplicated(subset=["ticker", "date", "article_id"])
    assert not dupes.any(), f"{dupes.sum()} duplicate (ticker, date, article_id) rows"

    # ── Each article should appear on exactly news_window trading dates ───────
    window_sizes = df.groupby("article_id")["date"].nunique()
    assert (window_sizes == cfg_train["news_window"]).all(), \
        "Some articles do not map to exactly 7 trading dates"

    # ── Boundary rule checks on synthetic examples ─────────────────────────────
    boundary_rows = [
        {
            "ticker": "AAPL",
            "publish_datetime": pd.Timestamp("2025-01-06T21:00:00Z"),
            "headline": "Exactly four PM ET headline for boundary check",
            "url": "boundary_same_day",
        },
        {
            "ticker": "AAPL",
            "publish_datetime": pd.Timestamp("2025-01-06T21:00:01Z"),
            "headline": "After four PM ET headline for next day check",
            "url": "boundary_next_day",
        },
    ]
    boundary_df = pd.DataFrame(boundary_rows)
    boundary_out = clean(
        boundary_df,
        news_window=cfg_train["news_window"],
        min_headline_len=cfg_pre.get("min_headline_len", 10),
        garbled_ratio_threshold=cfg_pre.get("garbled_non_ascii_ratio", 0.30),
        cutoff_hour_et=cfg_pre.get("cutoff_hour_et", 16),
    )
    boundary_dates = boundary_out.groupby("headline")["date"].min().to_dict()
    assert boundary_dates["Exactly four PM ET headline for boundary check"] == "2025-01-06", \
        "Exactly 4:00 PM ET should stay on the same trading day"
    assert boundary_dates["After four PM ET headline for next day check"] == "2025-01-07", \
        "After 4:00 PM ET should move to the next trading day"

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  OK — {len(df):,} rows saved to {out_path}")
    print(f"       tickers: {sorted(df['ticker'].unique())}")
    print(f"       date range: {df['date'].min()} ~ {df['date'].max()}")


def check_step_b(mock_bcd: bool):
    print("\n[Step B] Checking news_article_embeddings.parquet ...")
    path = ROOT / CONFIG["data"]["article_embeddings"]
    if mock_bcd or not path.exists():
        make_mock_article_embeddings(path)
    df = pd.read_parquet(path)
    assert all(c in df.columns for c in FEAT_COLS), "Missing feat_* columns"
    assert df.shape[1] == 3 + 771, f"Column count wrong: {df.shape[1]}"
    print(f"  OK — {len(df):,} rows, shape: {df.shape}")


def check_step_c(mock_bcd: bool):
    print("\n[Step C] Checking news_features.parquet ...")
    path = ROOT / CONFIG["data"]["news_features"]
    if mock_bcd or not path.exists():
        make_mock_news_features(path)
    df = pd.read_parquet(path)

    # Support both flat columns and MultiIndex (aggregator.py outputs MultiIndex;
    # if reading back as MultiIndex, reset to flat so dataset.py can use it)
    if df.index.names == ["ticker", "date"]:
        df = df.reset_index()

    assert "ticker" in df.columns and "date" in df.columns, \
        f"Missing ticker/date columns; columns={df.columns.tolist()}"
    feat_cols_present = [c for c in df.columns if c.startswith("feat_")]
    assert len(feat_cols_present) == 771, f"Feature dim wrong: {len(feat_cols_present)}"
    print(f"  OK — {len(df):,} (ticker, date) rows, {len(feat_cols_present)} feat cols")


def check_step_d(mock_bcd: bool):
    print("\n[Step D] Checking Dataset + Model can forward pass ...")
    feat_path  = ROOT / CONFIG["data"]["news_features"]
    label_path = ROOT / CONFIG["data"]["labels"]

    if mock_bcd or not label_path.exists():
        make_mock_labels(label_path)
    if mock_bcd or not feat_path.exists():
        make_mock_news_features(feat_path)

    # Flatten MultiIndex if needed before passing to dataset
    df_feat = pd.read_parquet(feat_path)
    if df_feat.index.names == ["ticker", "date"]:
        df_feat = df_feat.reset_index()
        df_feat.to_parquet(feat_path, index=False)

    from src.news.dataset import NewsFeatureDataset
    from src.news.model import NewsUnimodalBaseline

    ds = NewsFeatureDataset(str(feat_path), str(label_path), split="train")
    assert len(ds) > 0, "Dataset is empty"
    feat, label = ds[0]
    assert feat.shape == (771,), f"feat shape wrong: {feat.shape}"

    model = NewsUnimodalBaseline()
    with torch.no_grad():
        prob = model(feat.unsqueeze(0))
    assert prob.shape == (1, 1), f"model output shape wrong: {prob.shape}"
    print(f"  OK — Dataset: {len(ds)} samples, model output: {prob.shape}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock-bcd", action="store_true",
        help="Force mock data for Steps B/C/D (Step A always uses real CSVs)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("QianTick News Module — End-to-End Integration Test")
    print("=" * 55)

    check_step_a()
    check_step_b(args.mock_bcd)
    check_step_c(args.mock_bcd)
    check_step_d(args.mock_bcd)

    print("\n" + "=" * 55)
    print("ALL STEPS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
