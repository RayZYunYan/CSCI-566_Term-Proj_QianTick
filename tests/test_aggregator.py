"""
Unit Tests for NewsFeatureAggregator (子任务 C)
------------------------------------------------
用 torch.randn mock 独立测试，无需真实数据。

运行方式：
    pytest tests/test_aggregator.py -v
"""

import pytest
import torch
import pandas as pd
import numpy as np
from src.news.aggregator import NewsFeatureAggregator, extract_features


FEAT_DIM = 771


class TestNewsFeatureAggregator:

    @pytest.fixture(params=["mean", "attention"])
    def agg(self, request):
        return NewsFeatureAggregator(mode=request.param, feature_dim=FEAT_DIM)

    def test_output_shape_normal(self, agg):
        """正常输入（多篇文章）输出应为 (771,)"""
        embeddings = torch.randn(7, FEAT_DIM)
        out = agg(embeddings)
        assert out.shape == (FEAT_DIM,), f"Expected ({FEAT_DIM},), got {out.shape}"

    def test_output_shape_single_article(self, agg):
        """单篇文章输入，输出应为 (771,)"""
        embeddings = torch.randn(1, FEAT_DIM)
        out = agg(embeddings)
        assert out.shape == (FEAT_DIM,)

    def test_zero_articles(self, agg):
        """零篇文章时，输出应为全零向量 (771,)"""
        embeddings = torch.zeros(0, FEAT_DIM)
        out = agg(embeddings)
        assert out.shape == (FEAT_DIM,)
        assert torch.all(out == 0), "零文章时输出应为全零"

    def test_max_articles_truncation(self, agg):
        """超过 50 篇时，应截断为最新 50 篇，输出仍为 (771,)"""
        embeddings = torch.randn(80, FEAT_DIM)
        out = agg(embeddings)
        assert out.shape == (FEAT_DIM,)

    def test_output_dtype(self, agg):
        """输出应为 float32"""
        embeddings = torch.randn(5, FEAT_DIM)
        out = agg(embeddings)
        assert out.dtype == torch.float32

    def test_mean_mode_correctness(self):
        """mean 模式：输出应等于输入的均值"""
        agg = NewsFeatureAggregator(mode="mean", feature_dim=FEAT_DIM)
        embeddings = torch.randn(4, FEAT_DIM)
        out = agg(embeddings)
        expected = embeddings.mean(dim=0)
        assert torch.allclose(out, expected, atol=1e-6), "mean 聚合结果不正确"

    def test_truncation_uses_latest_50(self):
        """超过 50 篇时仅保留最新 50 篇"""
        agg = NewsFeatureAggregator(mode="mean", feature_dim=FEAT_DIM)
        embeddings = torch.arange(80 * FEAT_DIM, dtype=torch.float32).reshape(80, FEAT_DIM)
        out = agg(embeddings)
        expected = embeddings[-50:].mean(dim=0)
        assert torch.allclose(out, expected, atol=1e-6)


def test_extract_features_with_label_coverage(tmp_path):
    feat_cols = [f"feat_{i}" for i in range(FEAT_DIM)]

    article_rows = []
    for i, date in enumerate(["2024-01-02", "2024-01-02", "2024-01-03"]):
        row = {"ticker": "AAPL", "date": date, "article_id": f"id_{i}"}
        row.update({c: float(i + 1) for c in feat_cols})
        article_rows.append(row)
    article_df = pd.DataFrame(article_rows)

    labels_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2024-01-02", "label": 1},
            {"ticker": "AAPL", "date": "2024-01-03", "label": 0},
            {"ticker": "AAPL", "date": "2024-01-04", "label": 1},  # no article => zero vector
        ]
    )

    article_path = tmp_path / "article_embeddings.parquet"
    labels_path = tmp_path / "labels.parquet"
    article_df.to_parquet(article_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    config = {
        "data": {
            "article_embeddings": str(article_path),
            "labels": str(labels_path),
            "news_features": str(tmp_path / "news_features.parquet"),
        },
        "model": {"feature_dim": FEAT_DIM, "aggregation": "mean"},
        "tickers": ["AAPL"],
    }

    out = extract_features(config)
    assert set(["ticker", "date"]).issubset(out.columns)
    assert len([c for c in out.columns if c.startswith("feat_")]) == FEAT_DIM
    assert len(out) == 3

    row_0202 = out[(out["ticker"] == "AAPL") & (out["date"] == "2024-01-02")].iloc[0]
    # mean([1, 2]) = 1.5
    assert np.isclose(float(row_0202["feat_0"]), 1.5)

    row_0204 = out[(out["ticker"] == "AAPL") & (out["date"] == "2024-01-04")].iloc[0]
    assert np.isclose(float(row_0204["feat_0"]), 0.0)
