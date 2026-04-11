"""
Fusion Model (Cross-Attention)
-------------------------------
架构：
    新闻特征 (771维)  → 线性投影 → 256维  ┐
    价格特征 (192维)  → 线性投影 → 256维  ┘→ Cross-Attention → 256维 → Sigmoid → P(涨)

Cross-Attention 设计：
    - Query: 价格特征（关注新闻中哪些信息与价格走势相关）
    - Key/Value: 新闻特征
    - 也可双向：price→news 和 news→price 各一个头，再 concat
"""

import torch
import torch.nn as nn


FUSION_DIM = 256  # 融合后的统一维度


class CrossAttentionFusion(nn.Module):
    """
    单向 Cross-Attention：price 作为 Query，news 作为 Key/Value。

    TODO: 可以扩展为双向 Cross-Attention（两个方向各自 attention 后 concat）
    """

    def __init__(self, news_dim: int = 771, price_dim: int = 192,
                 fusion_dim: int = FUSION_DIM, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.news_proj  = nn.Linear(news_dim, fusion_dim)
        self.price_proj = nn.Linear(price_dim, fusion_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.output_dim = fusion_dim

    def forward(self, news_feat: torch.Tensor, price_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            news_feat:  (B, 771)
            price_feat: (B, 192)
        Returns:
            fused: (B, 256)
        """
        news_emb  = self.news_proj(news_feat).unsqueeze(1)   # (B, 1, 256)
        price_emb = self.price_proj(price_feat).unsqueeze(1) # (B, 1, 256)

        # price 作为 Q，news 作为 K/V
        attn_out, _ = self.cross_attn(
            query=price_emb,
            key=news_emb,
            value=news_emb,
        )  # (B, 1, 256)

        fused = self.norm(attn_out.squeeze(1) + price_emb.squeeze(1))  # residual
        return fused  # (B, 256)


class QianTickModel(nn.Module):
    """
    完整的 QianTick 多模态融合模型。
    """

    def __init__(self, news_dim: int = 771, price_dim: int = 192,
                 fusion_dim: int = FUSION_DIM):
        super().__init__()
        self.fusion = CrossAttentionFusion(news_dim, price_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, news_feat: torch.Tensor, price_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            news_feat:  (B, 771)
            price_feat: (B, 192)
        Returns:
            prob: (B, 1)  P(涨)
        """
        fused = self.fusion(news_feat, price_feat)  # (B, 256)
        return self.classifier(fused)               # (B, 1)
