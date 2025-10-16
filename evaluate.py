from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch

from src.utils import (
    build_loader_from_config,
    build_model,
    evaluate_model,
    get_logger,
    load_checkpoint,
)

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 TAnomalyCLIP 模型")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset_ucf.yaml"),
        help="数据集配置文件路径（YAML）。",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="覆盖配置中的注释文件路径（可选）。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="覆盖配置中的数据根目录（可选）。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="待评估模型的 checkpoint 路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备（如 cuda, cpu, cuda:0）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="评估批量大小（默认使用配置文件中的值）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 线程数（默认使用配置文件中的值）。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="计算视频级 Precision/Recall/F1 时的分数阈值。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval_metrics.json"),
        help="评估指标输出文件（JSON）。",
    )
    parser.add_argument(
        "--save-scores",
        type=Path,
        default=None,
        help="可选，保存详细分数（JSON 文件）。",
    )
    return parser.parse_args()


def format_metrics(metrics: Dict[str, float]) -> Dict[str, float | None]:
    formatted: Dict[str, float | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value):
                formatted[key] = None
            else:
                formatted[key] = round(value, 6)
        else:
            formatted[key] = value
    return formatted


def main() -> None:
    args = parse_args()

    desired_device = args.device
    if "cuda" in desired_device and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，自动切换至 CPU。")
        desired_device = "cpu"
    device = torch.device(desired_device)
    logger.info("使用设备：%s", device)

    data_loader = build_loader_from_config(
        args.dataset_config,
        override_annotation=args.annotation,
        override_data_root=args.data_root,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    logger.info("评估样本数量：%d", len(data_loader.dataset))

    model = build_model(device=device, freeze_backbone=True)
    load_checkpoint(model, Path(args.checkpoint))

    metrics, details = evaluate_model(
        model,
        data_loader,
        device=device,
        threshold=args.threshold,
    )

    safe_metrics = format_metrics(metrics)
    logger.info("评估指标：%s", json.dumps(safe_metrics, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(safe_metrics, f, indent=2, ensure_ascii=False)
    logger.info("已将指标写入：%s", args.output)

    if args.save_scores:
        score_path = Path(args.save_scores)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info("已保存详细分数至：%s", score_path)


if __name__ == "__main__":
    main()
