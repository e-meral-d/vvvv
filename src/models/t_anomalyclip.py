from __future__ import annotations

from typing import Dict, List, Optional

import open_clip
import torch
from torch import nn

from .ski_module import SKIModule, build_behavior_priors
from .temporal_encoder import TemporalEncoder


class PromptAnomalyHead(nn.Module):
    """
    基于可学习提示的异常检测头。
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        base_prompts: torch.Tensor,
        *,
        logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if base_prompts.shape != (2, embed_dim):
            raise ValueError("base_prompts 需为形状 (2, embed_dim) 的张量。")

        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        self.register_buffer("base_prompts", base_prompts)
        self.prompt_delta = nn.Parameter(torch.zeros_like(base_prompts))
        self.logit_scale = logit_scale

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError(f"输入应为三维张量 (B, T, D)，实际形状为 {tuple(features.shape)}。")

        projected = self.feature_proj(features)
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-6)

        prompts = self.base_prompts + self.prompt_delta
        prompts = prompts / (prompts.norm(dim=-1, keepdim=True) + 1e-6)
        normal_prompt, abnormal_prompt = prompts[0], prompts[1]

        normal_sim = torch.einsum("btd,d->bt", projected, normal_prompt)
        abnormal_sim = torch.einsum("btd,d->bt", projected, abnormal_prompt)
        frame_logits = (abnormal_sim - normal_sim) * self.logit_scale
        video_logits = frame_logits.max(dim=1).values

        return {
            "frame_logits": frame_logits,
            "video_logits": video_logits,
            "normal_similarity": normal_sim,
            "abnormal_similarity": abnormal_sim,
            "prompts": prompts,
        }


class TAnomalyCLIP(nn.Module):
    """
    零样本视频异常检测模型。
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Optional[torch.device] = None,
        freeze_backbone: bool = True,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
        prompt_logit_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        self.clip_model.eval()
        self.clip_model.to(self.device)

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.visual_dim = getattr(self.clip_model.visual, "output_dim")
        text_projection = getattr(self.clip_model, "text_projection")
        if text_projection is None:
            raise RuntimeError("CLIP 模型缺少 text_projection，无法对齐多模态空间。")
        self.text_dim = text_projection.shape[1]
        if self.text_dim != self.visual_dim:
            raise RuntimeError("视觉与文本嵌入维度不一致，当前实现要求二者一致。")

        self.temporal_encoder = TemporalEncoder(
            embed_dim=self.visual_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
        )

        behavior_embeddings = self._encode_behavior_priors().to(self.device)
        self.ski_module = SKIModule(behavior_embeddings)
        ski_output_dim = self.temporal_encoder.output_dim + behavior_embeddings.shape[1]

        base_prompts = self._encode_base_prompts().to(self.device)
        if prompt_logit_scale is not None:
            logit_scale = prompt_logit_scale
        else:
            logit_scale = float(self.clip_model.logit_scale.detach().cpu().exp().item())
        self.det_head = PromptAnomalyHead(
            feature_dim=ski_output_dim,
            embed_dim=self.text_dim,
            base_prompts=base_prompts,
            logit_scale=logit_scale,
        )

    def forward(
        self,
        video_clip: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if video_clip.dim() != 5:
            raise ValueError("输入视频应为五维张量 (B, C, T, H, W)。")

        b, c, t, h, w = video_clip.shape
        frames = video_clip.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        frames = frames.to(self.device)
        with torch.set_grad_enabled(self._vision_requires_grad):
            frame_features = self.clip_model.encode_image(frames)
        frame_features = frame_features.view(b, t, -1)

        temporal_features = self.temporal_encoder(frame_features, padding_mask=padding_mask)
        enriched, knowledge, weights = self.ski_module(temporal_features)
        outputs = self.det_head(enriched)

        if return_intermediate:
            outputs.update(
                {
                    "temporal_features": temporal_features,
                    "ski_knowledge": knowledge,
                    "ski_weights": weights,
                }
            )
        return outputs

    @property
    def _vision_requires_grad(self) -> bool:
        return any(param.requires_grad for param in self.clip_model.visual.parameters())

    def _encode_behavior_priors(self) -> torch.Tensor:
        priors = build_behavior_priors()
        phrases = priors["normal"] + priors["abnormal"]
        tokenized = open_clip.tokenize(phrases)
        tokenized = tokenized.to(self.device)
        with torch.no_grad():
            embeddings = self.clip_model.encode_text(tokenized)
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        return embeddings

    def _encode_base_prompts(self) -> torch.Tensor:
        prompts = [
            "a description of a normal interaction between people",
            "a description of an abnormal or violent interaction between people",
        ]
        tokenized = open_clip.tokenize(prompts)
        tokenized = tokenized.to(self.device)
        with torch.no_grad():
            embeddings = self.clip_model.encode_text(tokenized)
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        return embeddings

    def freeze_visual_backbone(self) -> None:
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False

    def unfreeze_visual_backbone(self) -> None:
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
