from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_lines(path: Path) -> List[str]:
    entries: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def resolve_image_path(dataset_root: Path, entry: str) -> Path:
    raw = Path(entry)

    candidates: List[Path] = []
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
    matches = [p for p in dataset_root.rglob(stem + ".*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(f"Khong resolve duoc image path cho entry: {entry}")


def infer_label_path(dataset_root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")

    try:
        rel = image_path.relative_to(dataset_root)
        return (dataset_root / "labels" / rel).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")


def parse_yolo_label_file(
    label_path: Path,
    image_w: int,
    image_h: int,
    class_id_map: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes: List[List[float]] = []
    labels: List[int] = []

    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            if class_id not in class_id_map:
                continue

            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2.0) * image_w
            y1 = (yc - bh / 2.0) * image_h
            x2 = (xc + bw / 2.0) * image_w
            y2 = (yc + bh / 2.0) * image_h

            x1 = max(0.0, min(float(image_w - 1), x1))
            y1 = max(0.0, min(float(image_h - 1), y1))
            x2 = max(0.0, min(float(image_w - 1), x2))
            y2 = max(0.0, min(float(image_h - 1), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id_map[class_id])

    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        label_tensor = torch.zeros((0,), dtype=torch.int64)

    return box_tensor, label_tensor


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split_list: str | Path,
        class_names: Sequence[str],
        train: bool = True,
        random_hflip: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split_list = Path(split_list).expanduser().resolve()
        self.class_names = list(class_names)
        self.train = train
        self.random_hflip = random_hflip

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Khong tim thay dataset root: {self.dataset_root}")
        if not self.split_list.exists():
            raise FileNotFoundError(f"Khong tim thay split file: {self.split_list}")

        self.class_id_map = {idx: idx + 1 for idx in range(len(self.class_names))}
        self.entries = read_lines(self.split_list)
        self.image_paths = [resolve_image_path(self.dataset_root, entry) for entry in self.entries]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size

        label_path = infer_label_path(self.dataset_root, image_path)
        boxes, labels = parse_yolo_label_file(
            label_path=label_path,
            image_w=image_w,
            image_h=image_h,
            class_id_map=self.class_id_map,
        )

        image_tensor = F.to_tensor(image)

        if self.train and self.random_hflip and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = float(image_w) - x2
                boxes[:, 2] = float(image_w) - x1

        area = torch.zeros((boxes.shape[0],), dtype=torch.float32)
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }

        return image_tensor, target, str(image_path)


def detection_collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)
