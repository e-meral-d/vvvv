from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

BEHAVIOR_PRIORS: Dict[str, List[str]] = {
    "normal": [
        "smooth movement",
        "walking at a steady pace",
        "organized crowd flow",
    ],
    "abnormal": [
        "erratic movement",
        "sudden acceleration",
        "fast chaotic motion",
        "physical struggle",
        "chasing and fleeing",
        "aggressive close contact",
        "falling motion",
        "aggressive posture",
    ],
}


def build_behavior_priors() -> Dict[str, List[str]]:
    return BEHAVIOR_PRIORS.copy()


class SKIModule(nn.Module):
    """
    语义注入模块：将行为先验编码向量融合到时序特征中。
    """

    def __init__(
        self,
        prior_embeddings: torch.Tensor,
        *,
        temperature: float = 1.0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        if prior_embeddings.dim() != 2:
            raise ValueError(
                f"先验嵌入应为二维张量 (N, D)，实际形状为 {tuple(prior_embeddings.shape)}。"
            )
        self.concat = concat
        self.temperature = temperature
        self.register_buffer("prior_embeddings", prior_embeddings)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError(f"输入应为三维张量 (B, T, D)，实际形状为 {tuple(features.shape)}。")

        priors = self.prior_embeddings.to(features.dtype)
        logits = torch.matmul(features, priors.t()) / max(self.temperature, 1e-6)
        weights = torch.sigmoid(self.scale * logits)
        knowledge = torch.matmul(weights, priors)
        if self.concat:
            enriched = torch.cat([features, knowledge], dim=-1)
        else:
            enriched = features + knowledge
        return enriched, knowledge, weights
