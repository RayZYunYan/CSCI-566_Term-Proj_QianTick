"""
Unit Tests for FinBERTEncoder (子任务 B)
----------------------------------------
用假 headline 独立测试，无需真实数据。

运行方式：
    pytest tests/test_encoder.py -v
"""

import pytest
import torch
from src.news.encoder import FinBERTEncoder


SAMPLE_HEADLINES = [
    "Apple beats Q3 earnings expectations",
    "NVIDIA announces new AI chip architecture",
    "Tesla recalls 500,000 vehicles over safety concerns",
    "Microsoft acquires gaming studio for $2B",
    "Market drops amid inflation fears",
]


class TestFinBERTEncoder:

    @pytest.fixture(scope="class")
    def encoder(self):
        return FinBERTEncoder(device="cpu")

    def test_output_shape_single(self, encoder):
        """单条文本输出形状应为 (1, 771)"""
        out = encoder.encode(["Apple reports record revenue"])
        assert out.shape == (1, 771), f"Expected (1, 771), got {out.shape}"

    def test_output_shape_batch(self, encoder):
        """批量文本输出形状应为 (N, 771)"""
        out = encoder.encode(SAMPLE_HEADLINES)
        assert out.shape == (len(SAMPLE_HEADLINES), 771)

    def test_sentiment_is_probability(self, encoder):
        """后 3 维应为概率（每行和为 1，值在 [0,1] 之间）"""
        out = encoder.encode(SAMPLE_HEADLINES)
        sentiment = out[:, 768:]
        assert sentiment.shape == (len(SAMPLE_HEADLINES), 3)
        assert (sentiment >= 0).all() and (sentiment <= 1).all()
        assert torch.allclose(sentiment.sum(dim=-1), torch.ones(len(SAMPLE_HEADLINES)), atol=1e-5)

    def test_bert_weights_frozen(self, encoder):
        """所有 BERT 参数应为 frozen（requires_grad=False）"""
        for name, param in encoder.bert.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_output_dtype(self, encoder):
        """输出应为 float32"""
        out = encoder.encode(["test headline"])
        assert out.dtype == torch.float32

    def test_long_text_truncation(self, encoder):
        """超长文本（>512 tokens）应正常截断，不报错"""
        long_text = "word " * 600
        out = encoder.encode([long_text])
        assert out.shape == (1, 771)
