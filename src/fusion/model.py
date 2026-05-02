"""
Fusion Model (Cross-Attention + Dual-Task)
------------------------------------------
架构：
    新闻特征 (771维)  → 线性投影 → fusion_dim ┐
    价格特征 (320维)  → 线性投影 → fusion_dim ┘→ Cross-Attention → LayerNorm + Residual
                                                        ↓
                                            ┌───────────────────────┐
                                            │  cls head → P(up)     │
                                            │  reg head → return    │
                                            └───────────────────────┘

Cross-Attention: price 作为 Query，news 作为 Key/Value。
Dual-Task Loss:  L = α * BCE(cls) + (1-α) * Huber(reg)
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        news_dim: int = 771,
        price_dim: int = 320,
        fusion_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.news_proj  = nn.Linear(news_dim,  fusion_dim)
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
            news_feat:  (B, news_dim)
            price_feat: (B, price_dim)
        Returns:
            fused: (B, fusion_dim)
        """
        news_emb  = self.news_proj(news_feat).unsqueeze(1)   # (B, 1, fusion_dim)
        price_emb = self.price_proj(price_feat).unsqueeze(1) # (B, 1, fusion_dim)

        # price 作为 Q，news 作为 K/V
        attn_out, _ = self.cross_attn(
            query=price_emb,
            key=news_emb,
            value=news_emb,
        )  # (B, 1, fusion_dim)

        # Residual + LayerNorm
        fused = self.norm(attn_out.squeeze(1) + price_emb.squeeze(1))
        return fused  # (B, fusion_dim)


class QianTickModel(nn.Module):
    """
    完整的 QianTick 多模态融合模型，dual-task 输出。
    """

    def __init__(
        self,
        news_dim: int = 771,
        price_dim: int = 320,
        fusion_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion = CrossAttentionFusion(news_dim, price_dim, fusion_dim, nhead, dropout)

        # 分类头：预测 P(up)，输出 logit（训练时用 BCEWithLogitsLoss）
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 回归头：预测下一交易日收益率
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, news_feat: torch.Tensor, price_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logit: (B,)  — 用于 BCEWithLogitsLoss
            reg_pred:  (B,)  — 用于 HuberLoss
        """
        fused = self.fusion(news_feat, price_feat)       # (B, fusion_dim)
        cls_logit = self.cls_head(fused).squeeze(-1)     # (B,)
        reg_pred  = self.reg_head(fused).squeeze(-1)     # (B,)
        return cls_logit, reg_pred
