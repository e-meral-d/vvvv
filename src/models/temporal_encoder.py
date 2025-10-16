from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """
    使用 Transformer 编码帧级特征序列。
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_positions: int = 512,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        mlp_dim = mlp_dim or embed_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_positions, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"输入应为三维张量 (B, T, D)，实际形状为 {tuple(x.shape)}。")

        batch, length, _ = x.shape
        if length > self.max_positions:
            raise ValueError(
                f"序列长度 {length} 超过了最大支持长度 {self.max_positions}。"
            )

        pos_emb = self.positional_embedding[:, :length, :]
        hidden = x + pos_emb
        hidden = self.dropout(hidden)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.norm(encoded)

    @property
    def output_dim(self) -> int:
        return self.embed_dim
