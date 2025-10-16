from __future__ import annotations

import torch

from src.models import TAnomalyCLIP


def build_model(device: torch.device, *, freeze_backbone: bool = True) -> TAnomalyCLIP:
    model = TAnomalyCLIP(device=device, freeze_backbone=freeze_backbone)
    model.to(device)
    return model
