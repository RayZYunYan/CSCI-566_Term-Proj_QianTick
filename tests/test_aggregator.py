"""
Unit Tests for NewsFeatureAggregator (子任务 C)
------------------------------------------------
用 torch.randn mock 独立测试，无需真实数据。

运行方式：
    pytest tests/test_aggregator.py -v
"""

import pytest
import torch
from src.news.aggregator import NewsFeatureAggregator


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
