from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from torch.utils.data import DataLoader

from .dataset import VideoAnomalyDataset
from ..utils.logging import get_logger

logger = get_logger(__name__)


def build_dataloader(
    annotation_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    *,
    clip_length: int = 16,
    target_frame_rate: int = 8,
    stride: int = 8,
    image_size: int = 336,
    is_train: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    构建 PyTorch DataLoader。
    """
    annotation = load_annotation(annotation_path)
    dataset = VideoAnomalyDataset(
        annotation=annotation,
        data_root=data_root,
        clip_length=clip_length,
        target_frame_rate=target_frame_rate,
        stride=stride,
        image_size=image_size,
        is_train=is_train,
    )

    if shuffle is None:
        shuffle = is_train

    logger.info(
        "构建 DataLoader：batch_size=%d, num_workers=%d, shuffle=%s",
        batch_size,
        num_workers,
        shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )


def load_annotation(annotation_path: Path) -> Dict[str, Any]:
    """
    加载 YAML/JSON 注释文件。
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"注释文件不存在：{path}")

    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if suffix == ".json":
            return json.load(f)
        raise ValueError(f"不支持的注释文件格式：{suffix}")
