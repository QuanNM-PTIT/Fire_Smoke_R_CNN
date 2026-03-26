from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_label_flags(label_path: Path, fire_ids: set[int], smoke_ids: set[int]) -> Tuple[int, int]:
    fire = 0
    smoke = 0
    if not label_path.exists():
        return fire, smoke
    for raw in label_path.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        class_id = int(float(raw.split()[0]))
        if class_id in fire_ids:
            fire = 1
        if class_id in smoke_ids:
            smoke = 1
    return fire, smoke


def collect_split(dataset_root: Path, split_name: str, fire_ids: set[int], smoke_ids: set[int]) -> List[Dict]:
    images_dir = dataset_root / split_name / 'images'
    labels_dir = dataset_root / split_name / 'labels'
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f'Missing {images_dir} or {labels_dir}')

    records: List[Dict] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS or not img_path.is_file():
            continue
        label_path = labels_dir / f'{img_path.stem}.txt'
        fire, smoke = read_label_flags(label_path, fire_ids, smoke_ids)
        records.append({
            'path': str(img_path.relative_to(dataset_root)),
            'fire': fire,
            'smoke': smoke,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--output-dir', default='data_csv')
    ap.add_argument('--fire-class-ids', default='0')
    ap.add_argument('--smoke-class-ids', default='1')
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fire_ids = {int(x) for x in args.fire_class_ids.split(',') if x.strip()}
    smoke_ids = {int(x) for x in args.smoke_class_ids.split(',') if x.strip()}

    train_records = collect_split(dataset_root, 'train', fire_ids, smoke_ids)
    test_records = collect_split(dataset_root, 'test', fire_ids, smoke_ids)

    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    n_val = max(1, int(len(train_records) * args.val_ratio))
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    for name, records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        df = pd.DataFrame(records)
        df.to_csv(output_dir / f'{name}.csv', index=False)
        print(f'[DONE] {name}.csv: {len(df)} rows')
        print(df[['fire', 'smoke']].value_counts().sort_index())


if __name__ == '__main__':
    main()