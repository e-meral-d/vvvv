from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency at runtime
    yaml = None


NORMAL_DIR_NAMES = {
    "Training_Normal_Videos_Anomaly",
    "Testing_Normal_Videos_Anomaly",
}


@dataclass
class VideoEntry:
    """Container holding a single video annotation entry."""

    video_id: str
    relative_path: str
    label: str

    def to_dict(self) -> dict:
        return {
            "id": self.video_id,
            "path": self.relative_path,
            "label": self.label,
            "segments": [],
        }


def parse_split_file(split_path: Path, *, split_tag: str) -> List[VideoEntry]:
    entries: List[VideoEntry] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            rel_path = raw.replace("\\", "/")
            top_dir = rel_path.split("/", 1)[0]
            label = "normal" if top_dir in NORMAL_DIR_NAMES else "anomaly"

            # Ensure consistent video id naming across splits
            stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            video_id = f"{split_tag}_{stem}"

            entries.append(VideoEntry(video_id=video_id, relative_path=rel_path, label=label))
    return entries


def validate_paths(entries: List[VideoEntry], videos_root: Path) -> List[VideoEntry]:
    missing: List[VideoEntry] = []
    for entry in entries:
        abs_path = videos_root / entry.relative_path
        if not abs_path.exists():
            missing.append(entry)
    return missing


def dump_annotation(entries: List[VideoEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"videos": [entry.to_dict() for entry in entries]}

    if yaml is not None:
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        return

    # Fallback: write a minimal YAML-compatible structure manually.
    with output_path.open("w", encoding="utf-8") as f:
        f.write("videos:\n")
        for item in payload["videos"]:
            f.write("  - id: {}\n".format(item["id"]))
            f.write("    path: {}\n".format(item["path"]))
            f.write("    label: {}\n".format(item["label"]))
            f.write("    segments: []\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate UCF-Crime annotation files from official splits.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/UCF_Crimes/Anomaly_Detection_splits"),
        help="Directory containing Anomaly_Train.txt and Anomaly_Test.txt.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("data/UCF_Crimes/Videos"),
        help="Root directory containing class sub-folders with video files.",
    )
    parser.add_argument(
        "--train-split-file",
        type=str,
        default="Anomaly_Train.txt",
        help="Filename of the training split text file.",
    )
    parser.add_argument(
        "--test-split-file",
        type=str,
        default="Anomaly_Test.txt",
        help="Filename of the testing split text file.",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("configs/annotations/ucf_train.yaml"),
        help="Output path for the generated training annotation file.",
    )
    parser.add_argument(
        "--output-test",
        type=Path,
        default=Path("configs/annotations/ucf_test.yaml"),
        help="Output path for the generated testing annotation file.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only generate the training annotation file.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only generate the testing annotation file.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    splits_dir = args.splits_dir
    videos_root = args.videos_root

    if not splits_dir.exists():
        print(f"[ERROR] Splits directory does not exist: {splits_dir}", file=sys.stderr)
        return 1
    if not videos_root.exists():
        print(f"[ERROR] Videos root does not exist: {videos_root}", file=sys.stderr)
        return 1

    if not args.skip_train:
        train_split_path = splits_dir / args.train_split_file
        if not train_split_path.exists():
            print(f"[ERROR] Train split file not found: {train_split_path}", file=sys.stderr)
            return 1
        train_entries = parse_split_file(train_split_path, split_tag="train")
        missing_train = validate_paths(train_entries, videos_root)
        if missing_train:
            print(f"[WARNING] {len(missing_train)} training videos listed but not found under {videos_root}:")
            for entry in missing_train[:10]:
                print(f"  - {entry.relative_path}")
            if len(missing_train) > 10:
                print("  ...")
        dump_annotation(train_entries, args.output_train)
        print(f"[INFO] Wrote training annotation with {len(train_entries)} entries to {args.output_train}")

    if not args.skip_test:
        test_split_path = splits_dir / args.test_split_file
        if not test_split_path.exists():
            print(f"[ERROR] Test split file not found: {test_split_path}", file=sys.stderr)
            return 1
        test_entries = parse_split_file(test_split_path, split_tag="test")
        missing_test = validate_paths(test_entries, videos_root)
        if missing_test:
            print(f"[WARNING] {len(missing_test)} testing videos listed but not found under {videos_root}:")
            for entry in missing_test[:10]:
                print(f"  - {entry.relative_path}")
            if len(missing_test) > 10:
                print("  ...")
        dump_annotation(test_entries, args.output_test)
        print(f"[INFO] Wrote testing annotation with {len(test_entries)} entries to {args.output_test}")

    return 0


if __name__ == "__main__":
    parser = build_argparser()
    exit_code = main(parser.parse_args())
    sys.exit(exit_code)
