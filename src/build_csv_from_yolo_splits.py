\
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_lines(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def resolve_image_path(dataset_root: Path, entry: str) -> Path:
    raw = Path(entry)

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(dataset_root / raw)
        candidates.append(dataset_root / "images" / raw)
        for ext in IMAGE_EXTENSIONS:
            candidates.append(dataset_root / f"{entry}{ext}")
            candidates.append(dataset_root / "images" / f"{entry}{ext}")

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    stem = raw.stem
    matches = list(dataset_root.rglob(stem + ".*"))
    matches = [m for m in matches if m.suffix.lower() in IMAGE_EXTENSIONS]
    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(f"Không resolve được image path cho entry: {entry}")


def infer_label_path(dataset_root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")

    rel = image_path.relative_to(dataset_root)
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def parse_yolo_labels(label_path: Path, fire_class_ids: Sequence[int], smoke_class_ids: Sequence[int]) -> Tuple[int, int]:
    fire = 0
    smoke = 0

    if not label_path.exists():
        return fire, smoke

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(float(parts[0]))
        if class_id in fire_class_ids:
            fire = 1
        if class_id in smoke_class_ids:
            smoke = 1

    return fire, smoke


def build_records(
    dataset_root: Path,
    list_path: Path,
    split: str,
    fire_class_ids: Sequence[int],
    smoke_class_ids: Sequence[int],
) -> List[Dict]:
    records = []
    for entry in read_lines(list_path):
        image_path = resolve_image_path(dataset_root, entry)
        label_path = infer_label_path(dataset_root, image_path)
        fire, smoke = parse_yolo_labels(label_path, fire_class_ids, smoke_class_ids)
        records.append(
            {
                "path": str(image_path.relative_to(dataset_root)),
                "fire": fire,
                "smoke": smoke,
                "split": split,
            }
        )
    return records


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV files from YOLO labels and split txt files.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--train-list", type=str, required=True)
    parser.add_argument("--val-list", type=str, required=True)
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--fire-class-ids", type=str, default="0", help="Comma-separated ids. Default: 0")
    parser.add_argument("--smoke-class-ids", type=str, default="1", help="Comma-separated ids. Default: 1")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fire_ids = parse_int_list(args.fire_class_ids)
    smoke_ids = parse_int_list(args.smoke_class_ids)

    all_records = []
    all_records += build_records(dataset_root, Path(args.train_list), "train", fire_ids, smoke_ids)
    all_records += build_records(dataset_root, Path(args.val_list), "val", fire_ids, smoke_ids)
    all_records += build_records(dataset_root, Path(args.test_list), "test", fire_ids, smoke_ids)

    df = pd.DataFrame(all_records)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split][["path", "fire", "smoke"]]
        csv_path = output_dir / f"{split}.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"[DONE] Saved {csv_path} with {len(split_df)} rows")

    summary = df.groupby(["split", "fire", "smoke"]).size().reset_index(name="count")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
