"""
News Unimodal Baseline (子任务 D)
----------------------------------
771 维新闻特征 → MLP → P(涨)
用于消融实验，评估仅用新闻的预测效果。

FinBERTEncoder   → src/news/encoder.py     (子任务 B)
NewsFeatureAggregator → src/news/aggregator.py (子任务 C)
"""

import torch
import torch.nn as nn


class NewsUnimodalBaseline(nn.Module):
    """
    新闻线 Unimodal Baseline：771 维特征 → MLP → P(涨)

    架构：771 → 256 → ReLU → Dropout(0.3) → 64 → ReLU → 1 → Sigmoid
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
            #nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 771)
        Returns:
            prob: (B, 1)  涨的概率
        """
        return self.mlp(x)
