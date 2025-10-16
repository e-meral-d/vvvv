from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from .transforms import ClipTransform, build_default_transform
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Segment:
    """表示一个时间段，单位为秒。"""

    start: float
    end: float

    def overlaps(self, other: "Segment") -> bool:
        return not (self.end <= other.start or other.end <= self.start)

    def contains(self, time_point: float) -> bool:
        return self.start <= time_point <= self.end


@dataclass
class ClipInfo:
    """预先计算好的片段元信息。"""

    video_path: Path
    video_id: str
    start: float
    end: float
    label: int
    frame_labels: torch.Tensor
    fps: float
    clip_index: int


class VideoAnomalyDataset(Dataset):
    """
    将长视频划分为固定长度的Clip，并输出用于训练/评估的张量。

    预期注释文件格式（YAML/JSON 均可解析为字典）：
    {
        "videos": [
            {
                "id": "video_0001",
                "path": "relative/path/to/video.mp4",
                "label": "anomaly" | "normal",
                "fps": 25.0,  # 可选，缺省则自动探测
                "segments": [
                    {"start": 12.4, "end": 20.0},  # 仅在存在异常区间时需要
                    ...
                ]
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        annotation: Dict,
        data_root: Path,
        clip_length: int = 16,
        target_frame_rate: int = 8,
        stride: int = 8,
        image_size: int = 336,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_train: bool = True,
        min_clip_duration: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        self.target_frame_rate = target_frame_rate
        self.stride = stride
        self.image_size = image_size
        self.transform = transform or build_default_transform(image_size, is_train=is_train)
        self.is_train = is_train
        self.min_clip_duration = min_clip_duration or clip_length / target_frame_rate

        videos = annotation.get("videos", [])
        if not videos:
            raise ValueError("注释文件中未找到任何视频条目（键 `videos`）。")

        logger.info("正在构建数据集索引，共 %d 段视频。", len(videos))
        self.clips: List[ClipInfo] = []
        for video_meta in videos:
            clip_items = self._build_clips_for_video(video_meta)
            self.clips.extend(clip_items)

        if not self.clips:
            raise RuntimeError("未能从注释文件中生成任何Clip，请检查配置与数据路径。")

        logger.info("数据集共生成 %d 个Clip。", len(self.clips))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict:
        clip_info = self.clips[index]
        start, end = clip_info.start, clip_info.end
        video_path = clip_info.video_path.as_posix()

        frames, _, info = read_video(
            video_path,
            start_pts=start,
            end_pts=end,
            pts_unit="sec",
        )
        if frames.numel() == 0:
            raise RuntimeError(f"视频 {video_path} 在区间 [{start}, {end}] 内未读取到帧。")

        fps = info.get("video_fps", clip_info.fps) or clip_info.fps
        clip = self._resample_clip(frames, fps)
        clip = self.transform(clip) if self.transform else clip

        return {
            "video_id": clip_info.video_id,
            "clip_index": clip_info.clip_index,
            "start": start,
            "end": end,
            "frames": clip,  # C x T x H x W
            "clip_label": clip_info.label,
            "frame_labels": clip_info.frame_labels,
            "video_path": video_path,
        }

    def _build_clips_for_video(self, video_meta: Dict) -> List[ClipInfo]:
        video_rel_path = video_meta.get("path")
        if not video_rel_path:
            raise ValueError("注释条目缺少 `path` 字段。")
        video_path = (self.data_root / video_rel_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在：{video_path}")

        video_id = str(video_meta.get("id", video_path.stem))
        anomaly_segments = _parse_segments(video_meta.get("segments", []))
        base_label = 1 if (video_meta.get("label") == "anomaly" or anomaly_segments) else 0

        fps, duration = probe_video_stats(video_path)
        fps = video_meta.get("fps", fps)
        clip_duration = self.clip_length / self.target_frame_rate
        stride_duration = self.stride / self.target_frame_rate

        if duration < self.min_clip_duration:
            logger.warning("视频 %s 时长 %.2fs 过短，跳过。", video_path.name, duration)
            return []

        clip_infos: List[ClipInfo] = []
        num_clips = max(
            1,
            int(math.floor((duration - clip_duration) / stride_duration)) + 1,
        )

        for clip_idx in range(num_clips):
            start_time = clip_idx * stride_duration
            end_time = min(start_time + clip_duration, duration)
            if end_time - start_time < clip_duration and duration >= clip_duration:
                start_time = duration - clip_duration
                end_time = duration

            clip_label = base_label
            frame_labels = torch.zeros(self.clip_length, dtype=torch.float32)

            if anomaly_segments:
                clip_segment = Segment(start=start_time, end=end_time)
                clip_label = 1 if any(seg.overlaps(clip_segment) for seg in anomaly_segments) else 0
                frame_times = self._frame_time_stamps(start_time)
                for idx, ts in enumerate(frame_times):
                    if any(seg.contains(ts) for seg in anomaly_segments):
                        frame_labels[idx] = 1.0

            clip_infos.append(
                ClipInfo(
                    video_path=video_path,
                    video_id=video_id,
                    start=start_time,
                    end=end_time,
                    label=clip_label,
                    frame_labels=frame_labels,
                    fps=fps,
                    clip_index=clip_idx,
                )
            )

        return clip_infos

    def _frame_time_stamps(self, start_time: float) -> List[float]:
        """返回 clip 内每帧对应的时间戳（秒）。"""
        return [
            start_time + (i / self.target_frame_rate)
            for i in range(self.clip_length)
        ]

    def _resample_clip(self, frames: torch.Tensor, source_fps: float) -> torch.Tensor:
        """
        将原始帧序列重采样为固定长度的 clip，并输出形状为 CxTxHxW 的张量。
        """
        num_frames = frames.shape[0]
        if num_frames == 0:
            raise ValueError("输入帧序列为空。")

        if num_frames != self.clip_length:
            indices = torch.linspace(
                0, max(0, num_frames - 1), steps=self.clip_length, device=frames.device
            ).long()
            frames = frames.index_select(0, indices)

        clip = frames.permute(0, 3, 1, 2).float() / 255.0  # T, C, H, W
        return clip.permute(1, 0, 2, 3)  # C, T, H, W


def _parse_segments(raw_segments: Optional[Sequence]) -> List[Segment]:
    segments: List[Segment] = []
    for seg in raw_segments or []:
        if isinstance(seg, dict):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            start = float(seg[0])
            end = float(seg[1])
        else:
            raise ValueError(f"无法解析的段信息：{seg}")

        if end <= start:
            raise ValueError(f"无效的时间段：start={start}, end={end}")
        segments.append(Segment(start=start, end=end))
    return segments


def probe_video_stats(video_path: Path) -> Tuple[float, float]:
    """
    轻量地探测视频的帧率与时长。

    返回:
        fps: float, 视频帧率
        duration: float, 视频时长（秒）
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps is None or fps <= 0:
        fps = 30.0  # 兜底
    if frame_count is None or frame_count <= 0:
        raise RuntimeError(f"无法读取视频帧数：{video_path}")

    duration = frame_count / fps
    return float(fps), float(duration)
