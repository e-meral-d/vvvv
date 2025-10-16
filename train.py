from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import TAnomalyCLIP
from src.utils import (
    build_loader_from_config,
    build_model,
    compute_epoch_metrics,
    evaluate_model,
    get_logger,
    load_checkpoint,
    save_checkpoint,
)

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TAnomalyCLIP 模型")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset_ucf.yaml"),
        help="训练数据集配置文件路径（YAML）。",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="覆盖配置文件中的注释文件路径（可选）。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="覆盖配置文件中的数据根目录（可选）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="训练日志与模型权重的保存目录。",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数。")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减系数。")
    parser.add_argument(
        "--frame-loss-weight",
        type=float,
        default=1.0,
        help="帧级 BCE 损失的损失权重。",
    )
    parser.add_argument(
        "--video-loss-weight",
        type=float,
        default=1.0,
        help="视频级 BCE 损失的损失权重。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备（如 cuda, cpu, cuda:0）。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="启用混合精度训练 (torch.cuda.amp)。",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="恢复训练的模型权重路径（state_dict）。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="保存 checkpoint 的间隔（按 epoch 计）。",
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        default=None,
        help="验证集注释文件路径（可选）。若提供，将在每个 epoch 后评估一次。",
    )
    parser.add_argument(
        "--val-data-root",
        type=Path,
        default=None,
        help="验证集数据根目录（可选）。",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="训练时日志输出的步数间隔。",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.5,
        help="计算 Precision/Recall/F1 时的分数阈值。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: TAnomalyCLIP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frame_loss_fn: nn.Module,
    video_loss_fn: nn.Module,
    loss_weights: Tuple[float, float],
    *,
    log_interval: int = 20,
    scaler: torch.cuda.amp.GradScaler | None = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_steps = len(loader)
    start_time = time.time()

    frame_weight, video_weight = loss_weights

    running_loss = 0.0
    frame_targets, frame_scores = [], []
    video_targets, video_scores = [], []

    for step, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        frame_labels = batch["frame_labels"].to(device, non_blocking=True).float()
        clip_labels = batch["clip_label"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(frames)
            frame_logits = outputs["frame_logits"]
            video_logits = outputs["video_logits"]

            frame_loss = frame_loss_fn(frame_logits, frame_labels)
            video_loss = video_loss_fn(video_logits, clip_labels)
            loss = frame_weight * frame_loss + video_weight * video_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        frame_targets.extend(frame_labels.detach().cpu().reshape(-1).tolist())
        frame_scores.extend(frame_logits.detach().cpu().reshape(-1).tolist())
        video_targets.extend(clip_labels.detach().cpu().tolist())
        video_scores.extend(video_logits.detach().cpu().tolist())

        if step % log_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            logger.info(
                "Step [%d/%d] - loss: %.4f (frame: %.4f, video: %.4f) - %.2f it/s",
                step,
                total_steps,
                loss.item(),
                frame_loss.item(),
                video_loss.item(),
                step / max(elapsed, 1e-6),
            )

    avg_loss = running_loss / max(total_steps, 1)
    metrics = compute_epoch_metrics(
        frame_targets,
        frame_scores,
        video_targets,
        video_scores,
        threshold=threshold,
    )
    metrics["loss"] = avg_loss
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_device = args.device
    if "cuda" in desired_device and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，自动切换至 CPU。")
        desired_device = "cpu"
    device = torch.device(desired_device)
    logger.info("使用设备：%s", device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = build_loader_from_config(
        args.dataset_config,
        override_annotation=args.annotation,
        override_data_root=args.data_root,
        is_train=True,
    )
    logger.info("训练样本数量：%d", len(train_loader.dataset))
    val_loader = None
    if args.val_annotation is not None:
        val_loader = build_loader_from_config(
            args.dataset_config,
            override_annotation=args.val_annotation,
            override_data_root=args.val_data_root or args.data_root,
            is_train=False,
        )
        logger.info("验证样本数量：%d", len(val_loader.dataset))

    model = build_model(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, Path(args.resume), optimizer=optimizer)

    frame_loss_fn = nn.BCEWithLogitsLoss()
    video_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info("====== Epoch %d / %d ======", epoch, args.epochs)
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            frame_loss_fn=frame_loss_fn,
            video_loss_fn=video_loss_fn,
            loss_weights=(args.frame_loss_weight, args.video_loss_weight),
            log_interval=args.log_interval,
            scaler=scaler if device.type == "cuda" else None,
            threshold=args.metric_threshold,
        )
        logger.info(
            "Epoch %d train metrics: %s",
            epoch,
            json.dumps({k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v for k, v in metrics.items()}),
        )
        history.append({"epoch": epoch, "train": metrics})

        if val_loader is not None:
            val_metrics, _ = evaluate_model(
                model,
                val_loader,
                device=device,
                threshold=args.metric_threshold,
            )
            logger.info(
                "Epoch %d validation metrics: %s",
                epoch,
                json.dumps({k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v for k, v in val_metrics.items()}),
            )
            history[-1]["val"] = val_metrics

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, output_dir)

    summary_path = output_dir / "training_history.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info("训练完成，指标记录已写入：%s", summary_path)


if __name__ == "__main__":
    main()
