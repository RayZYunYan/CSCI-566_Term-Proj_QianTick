"""
News Model (FinBERT-based)
--------------------------
架构：
    新闻文本 → FinBERT → [CLS] 768维 → mean pooling over 7天 →
    拼接 3维情感概率 → 771维新闻特征向量

同时包含 unimodal baseline：
    771维 → MLP → Sigmoid → P(涨)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


FINBERT_MODEL = "ProsusAI/finbert"  # HuggingFace 模型名


class FinBERTEncoder(nn.Module):
    """
    对单条新闻文本输出 768维语义向量 + 3维情感概率（negative/neutral/positive）。

    TODO:
        - 确认是否需要 fine-tune FinBERT，或仅用作特征提取（frozen）
        - 处理文本长度截断（max_length=512）
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.bert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_emb:    (N, 768)  [CLS] token 向量
            sentiment:  (N, 3)   softmax 情感概率
        """
        # TODO: 实现 tokenize → forward → 提取 [CLS] 和情感
        raise NotImplementedError


class NewsFeatureAggregator(nn.Module):
    """
    将 7天内多条新闻的特征聚合为单个日级别特征向量 (771维)。

    TODO: 可以尝试 mean / attention-weighted 聚合
    """

    def forward(self, cls_embs: torch.Tensor, sentiments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embs:   (T, 768)  T条新闻的 [CLS] 向量
            sentiments: (T, 3)

        Returns:
            feature: (771,)  聚合后特征
        """
        # TODO: 实现聚合逻辑
        raise NotImplementedError


class NewsUnimodalBaseline(nn.Module):
    """
    新闻线 Unimodal Baseline：771维特征 → MLP → P(涨)
    用于消融实验，评估仅用新闻的预测效果。
    """

    def __init__(self, input_dim: int = 771, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 771)
        Returns:
            prob: (B, 1)  涨的概率
        """
        return self.mlp(x)
