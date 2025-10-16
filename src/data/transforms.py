from __future__ import annotations

import random
from typing import Optional

import torch

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class ClipTransform:
    """
    针对视频 Clip（CxTxHxW）的基础图像增强与归一化。
    """

    def __init__(self, image_size: int, is_train: bool = True, flip_prob: float = 0.5) -> None:
        self.image_size = image_size
        self.is_train = is_train
        self.flip_prob = flip_prob

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 4:
            raise ValueError(f"预期 clip 维度为 4，实际为 {clip.shape}")

        if self.is_train and random.random() < self.flip_prob:
            clip = torch.flip(clip, dims=(3,))

        clip = torch.nn.functional.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = CLIP_MEAN.view(3, 1, 1, 1).to(clip.device)
        std = CLIP_STD.view(3, 1, 1, 1).to(clip.device)
        clip = (clip - mean) / std
        return clip


def build_default_transform(image_size: int, is_train: bool = True) -> ClipTransform:
    return ClipTransform(image_size=image_size, is_train=is_train)
