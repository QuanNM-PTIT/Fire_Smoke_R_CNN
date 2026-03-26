\
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}

LABEL_PATTERNS = [
    (["fire_and_smoke", "both", "fire_smoke"], (1, 1)),
    (["only_fire", "fire_only"], (1, 0)),
    (["only_smoke", "smoke_only"], (0, 1)),
    (["none", "normal", "negative", "background", "no_fire_no_smoke"], (0, 0)),
]


def normalize_token(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


def infer_split(path: Path) -> Optional[str]:
    for part in path.parts:
        key = normalize_token(part)
        if key in SPLIT_ALIASES:
            return SPLIT_ALIASES[key]
    return None


def infer_labels(path: Path) -> Optional[Tuple[int, int]]:
    normalized_parts = [normalize_token(p) for p in path.parts]

    for part in reversed(normalized_parts):
        for patterns, labels in LABEL_PATTERNS:
            if part in patterns:
                return labels

    return None


def collect_records(dataset_root: Path) -> List[Dict]:
    records = []
    for image_path in dataset_root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        split = infer_split(image_path)
        labels = infer_labels(image_path)

        if split is None or labels is None:
            continue

        fire, smoke = labels
        records.append(
            {
                "path": str(image_path.relative_to(dataset_root)),
                "fire": fire,
                "smoke": smoke,
                "split": split,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test CSV files from class folder structure.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Where to save train.csv / val.csv / test.csv")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(dataset_root)
    if not records:
        raise RuntimeError(
            "Không tìm thấy ảnh hợp lệ. "
            "Hãy kiểm tra lại tên thư mục split (train/val/test) và class "
            "(only_fire, only_smoke, fire_and_smoke, none)."
        )

    df = pd.DataFrame(records).sort_values(["split", "path"]).reset_index(drop=True)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split][["path", "fire", "smoke"]]
        if len(split_df) == 0:
            print(f"[WARN] Split {split} rỗng.")
            continue
        csv_path = output_dir / f"{split}.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"[DONE] Saved {csv_path} with {len(split_df)} rows")

    summary = df.groupby(["split", "fire", "smoke"]).size().reset_index(name="count")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
