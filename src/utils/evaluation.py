from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

from .metrics import binary_auc, precision_recall_f1


def compute_epoch_metrics(
    frame_targets: Iterable[float],
    frame_scores: Iterable[float],
    video_targets: Iterable[float],
    video_scores: Iterable[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    frame_auc = binary_auc(frame_targets, frame_scores)
    video_auc = binary_auc(video_targets, video_scores)
    v_precision, v_recall, v_f1 = precision_recall_f1(video_targets, video_scores, threshold=threshold)
    return {
        "frame_auc": frame_auc,
        "video_auc": video_auc,
        "video_precision": v_precision,
        "video_recall": v_recall,
        "video_f1": v_f1,
        "threshold": threshold,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    model.eval()
    frame_targets: List[float] = []
    frame_scores: List[float] = []
    video_targets: List[float] = []
    video_scores: List[float] = []
    video_ids: List[str] = []
    clip_indices: List[int] = []
    clip_start: List[float] = []
    clip_end: List[float] = []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        frame_labels = batch["frame_labels"].to(device, non_blocking=True).float()
        clip_labels = batch["clip_label"].to(device, non_blocking=True).float()

        outputs = model(frames)
        frame_logits = outputs["frame_logits"]
        video_logits = outputs["video_logits"]

        frame_targets.extend(frame_labels.cpu().reshape(-1).tolist())
        frame_scores.extend(frame_logits.cpu().reshape(-1).tolist())
        video_targets.extend(clip_labels.cpu().tolist())
        video_scores.extend(video_logits.cpu().tolist())

        raw_ids = batch.get("video_id")
        if raw_ids is not None:
            if isinstance(raw_ids, list):
                video_ids.extend([str(v) for v in raw_ids])
            else:
                video_ids.extend([str(raw_ids)])
        raw_indices = batch.get("clip_index")
        if raw_indices is not None:
            if isinstance(raw_indices, torch.Tensor):
                clip_indices.extend(raw_indices.cpu().tolist())
            elif isinstance(raw_indices, list):
                clip_indices.extend([int(x) for x in raw_indices])
            else:
                clip_indices.append(int(raw_indices))
        starts = batch.get("start")
        if starts is not None:
            if isinstance(starts, torch.Tensor):
                clip_start.extend(starts.cpu().tolist())
            elif isinstance(starts, list):
                clip_start.extend([float(x) for x in starts])
            else:
                clip_start.append(float(starts))
        ends = batch.get("end")
        if ends is not None:
            if isinstance(ends, torch.Tensor):
                clip_end.extend(ends.cpu().tolist())
            elif isinstance(ends, list):
                clip_end.extend([float(x) for x in ends])
            else:
                clip_end.append(float(ends))

    metrics = compute_epoch_metrics(
        frame_targets,
        frame_scores,
        video_targets,
        video_scores,
        threshold=threshold,
    )
    details = {
        "frame_targets": frame_targets,
        "frame_scores": frame_scores,
        "video_targets": video_targets,
        "video_scores": video_scores,
        "video_ids": video_ids,
        "clip_indices": clip_indices,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }
    return metrics, details
