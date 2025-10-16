from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    logger.info("已保存模型权重：%s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = int(checkpoint.get("epoch", 0))
    logger.info("已从 %s 恢复模型（epoch=%d）。", checkpoint_path, epoch)
    return epoch
