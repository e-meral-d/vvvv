from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate
