from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from src.data import build_dataloader

from .config import load_yaml, resolve_path


def build_loader_from_config(
    config_path: Path,
    *,
    override_annotation: Optional[Path] = None,
    override_data_root: Optional[Path] = None,
    is_train: bool = True,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    config = load_yaml(config_path)

    annotation = override_annotation or config.get("annotation")
    if annotation is None:
        raise ValueError("数据配置缺少 `annotation` 字段。")

    data_root = override_data_root or config.get("data_root")
    if data_root is None:
        raise ValueError("数据配置缺少 `data_root` 字段。")

    return build_dataloader(
        annotation_path=resolve_path(annotation),
        data_root=resolve_path(data_root),
        batch_size=batch_size if batch_size is not None else int(config.get("batch_size", 4)),
        num_workers=num_workers if num_workers is not None else int(config.get("num_workers", 4)),
        clip_length=int(config.get("clip_length", 16)),
        target_frame_rate=int(config.get("target_frame_rate", 8)),
        stride=int(config.get("stride", 8)),
        image_size=int(config.get("image_size", 336)),
        is_train=is_train,
        shuffle=shuffle if shuffle is not None else bool(config.get("shuffle", is_train)),
    )
