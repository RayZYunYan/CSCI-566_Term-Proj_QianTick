"""
News Module End-to-End Integration Test (子任务 A)
--------------------------------------------------
验证 A → B → C → D 四个子任务能串联跑通。
每一步先检查真实文件，不存在则用 mock 数据替代。

负责人：子任务 A
运行方式：
    python scripts/news_e2e_test.py
    python scripts/news_e2e_test.py --mock   # 全部用 mock 数据
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import yaml

# 保证从项目根目录运行
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "news_config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

FEAT_COLS = [f"feat_{i}" for i in range(771)]
TICKERS   = ["AAPL", "MSFT", "NVDA"]
DATES     = pd.bdate_range("2023-01-01", "2025-06-01").strftime("%Y-%m-%d").tolist()


# ── Mock 数据生成工具 ──────────────────────────────────────

def make_mock_cleaned(path: Path):
    """生成 mock news_cleaned.parquet（子任务 A 的输出）"""
    rows = []
    for t in TICKERS:
        for i, d in enumerate(DATES[:50]):
            rows.append({
                "ticker": t, "date": d,
                "article_id": f"{t}_{d}_{i:04d}",
                "headline": f"Mock headline for {t} on {d}",
                "publish_datetime": pd.Timestamp(d),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  [mock] Created {path} ({len(rows)} rows)")


def make_mock_article_embeddings(path: Path):
    """生成 mock news_article_embeddings.parquet（子任务 B 的输出）"""
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
    """生成 mock news_features.parquet（子任务 C 的输出）"""
    rows = []
    for t in TICKERS:
        for d in DATES[:200]:
            row = {"ticker": t, "date": d}
            row.update({c: float(np.random.randn()) for c in FEAT_COLS})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).set_index(["ticker", "date"])
    df.to_parquet(path)
    print(f"  [mock] Created {path} ({len(df)} rows)")


def make_mock_labels(path: Path):
    """生成 mock labels.parquet"""
    rows = [{"ticker": t, "date": d, "label": int(np.random.randint(0, 2))}
            for t in TICKERS for d in DATES[:200]]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  [mock] Created {path} ({len(rows)} rows)")


# ── 各步骤验证 ────────────────────────────────────────────

def check_step_a(mock: bool):
    print("\n[Step A] 检查 news_cleaned.parquet ...")
    path = Path(CONFIG["data"]["cleaned"])
    if mock or not path.exists():
        make_mock_cleaned(path)
    df = pd.read_parquet(path)
    assert set(["ticker", "date", "article_id", "headline", "publish_datetime"]).issubset(df.columns), \
        f"schema 不符合：{df.columns.tolist()}"
    assert df["ticker"].str.isupper().all(), "ticker 存在小写"
    assert df["headline"].notna().all(), "headline 含空值"
    print(f"  OK — {len(df):,} 行，tickers: {sorted(df['ticker'].unique())}")


def check_step_b(mock: bool):
    print("\n[Step B] 检查 news_article_embeddings.parquet ...")
    path = Path(CONFIG["data"]["article_embeddings"])
    if mock or not path.exists():
        make_mock_article_embeddings(path)
    df = pd.read_parquet(path)
    assert all(c in df.columns for c in FEAT_COLS), "缺少 feat_* 列"
    assert df.shape[1] == 3 + 771, f"列数不对：{df.shape[1]}"
    print(f"  OK — {len(df):,} 行，shape: {df.shape}")


def check_step_c(mock: bool):
    print("\n[Step C] 检查 news_features.parquet ...")
    path = Path(CONFIG["data"]["news_features"])
    if mock or not path.exists():
        make_mock_news_features(path)
    df = pd.read_parquet(path)
    assert df.index.names == ["ticker", "date"], f"index 不对：{df.index.names}"
    assert df.shape[1] == 771, f"特征维度不对：{df.shape[1]}"
    print(f"  OK — {len(df):,} (ticker, date) 对，shape: {df.shape}")


def check_step_d(mock: bool):
    print("\n[Step D] 检查 Dataset + Model 可跑通 ...")
    feat_path  = Path(CONFIG["data"]["news_features"])
    label_path = Path(CONFIG["data"]["labels"])

    if mock or not label_path.exists():
        make_mock_labels(label_path)
    if mock or not feat_path.exists():
        make_mock_news_features(feat_path)

    from src.news.dataset import NewsFeatureDataset
    from src.news.model import NewsUnimodalBaseline

    ds = NewsFeatureDataset(str(feat_path), str(label_path), split="train")
    assert len(ds) > 0, "Dataset 为空"
    feat, label = ds[0]
    assert feat.shape == (771,), f"feat shape 不对：{feat.shape}"

    model = NewsUnimodalBaseline()
    batch = feat.unsqueeze(0)  # (1, 771)
    with torch.no_grad():
        prob = model(batch)
    assert prob.shape == (1, 1), f"model 输出 shape 不对：{prob.shape}"
    print(f"  OK — Dataset: {len(ds)} 样本，Model 输出: {prob.shape}")


# ── 主入口 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="强制使用 mock 数据")
    args = parser.parse_args()

    print("=" * 50)
    print("QianTick News Module — End-to-End Integration Test")
    print("=" * 50)

    check_step_a(args.mock)
    check_step_b(args.mock)
    check_step_c(args.mock)
    check_step_d(args.mock)

    print("\n" + "=" * 50)
    print("ALL STEPS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
