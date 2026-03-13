"""
Price Model (LSTM / Transformer)
----------------------------------
架构：
    价格序列 (30天 OHLCV) → LSTM/Transformer → 128维价格特征
    技术指标              → MLP             → 64维指标特征
    拼接 → 192维 → 最终输出

同时包含 unimodal baseline：
    192维 → Linear → Sigmoid → P(涨)
"""

import torch
import torch.nn as nn


class LSTMPriceEncoder(nn.Module):
    """
    LSTM 编码器：(B, T, input_dim) → (B, 128)

    TODO: 可选替换为 Transformer encoder
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)  T=30
        Returns:
            feat: (B, 128)  最后时间步的隐状态
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # 最后一层的最后时间步


class TransformerPriceEncoder(nn.Module):
    """
    Transformer 编码器替代方案：(B, T, input_dim) → (B, 128)

    TODO: 实现完整的 Transformer encoder（含 positional encoding）
    """

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.transformer(x)         # (B, T, d_model)
        return x[:, -1, :]              # 取最后时间步 (B, d_model)


class TechIndicatorMLP(nn.Module):
    """技术指标 MLP 编码器 → 64维"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PriceUnimodalBaseline(nn.Module):
    """
    价格线 Unimodal Baseline：LSTM特征 + 技术指标特征 → P(涨)
    用于消融实验。
    """

    def __init__(self, ohlcv_dim: int, tech_dim: int, use_transformer: bool = False):
        super().__init__()
        if use_transformer:
            self.price_encoder = TransformerPriceEncoder(input_dim=ohlcv_dim)
        else:
            self.price_encoder = LSTMPriceEncoder(input_dim=ohlcv_dim)

        self.tech_encoder = TechIndicatorMLP(input_dim=tech_dim)

        fused_dim = self.price_encoder.output_dim + self.tech_encoder.output_dim  # 192
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq: torch.Tensor, tech: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:  (B, 30, ohlcv_dim)
            tech: (B, tech_dim)
        Returns:
            prob: (B, 1)
        """
        price_feat = self.price_encoder(seq)
        tech_feat  = self.tech_encoder(tech)
        fused = torch.cat([price_feat, tech_feat], dim=-1)
        return self.classifier(fused)
