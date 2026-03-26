\
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ["fire", "smoke"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name: str = "resnet18", pretrained: bool = True, num_outputs: int = 2) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model

    if model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_outputs)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    eps = 1e-6
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    pos_weight = (negatives + eps) / (positives + eps)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]
) -> Dict[str, float]:
    thresholds_arr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    y_pred = (y_prob >= thresholds_arr).astype(np.int32)
    y_true = y_true.astype(np.int32)

    metrics: Dict[str, float] = {}
    f1_values: List[float] = []

    for i, name in enumerate(CLASS_NAMES):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        tn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1
        metrics[f"{name}_accuracy"] = accuracy
        metrics[f"{name}_tp"] = tp
        metrics[f"{name}_fp"] = fp
        metrics[f"{name}_fn"] = fn
        metrics[f"{name}_tn"] = tn
        f1_values.append(f1)

    exact_match = float((y_pred == y_true).all(axis=1).mean())
    metrics["exact_match_accuracy"] = exact_match
    metrics["macro_f1"] = float(np.mean(f1_values))
    return metrics


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[List[float], Dict[str, float]]:
    candidate_thresholds = np.arange(0.10, 0.95, 0.05)
    best_thresholds: List[float] = []

    for i in range(y_true.shape[1]):
        best_thr = 0.5
        best_f1 = -1.0
        for thr in candidate_thresholds:
            preds = (y_prob[:, i] >= thr).astype(np.int32)
            truth = y_true[:, i].astype(np.int32)
            tp = int(((preds == 1) & (truth == 1)).sum())
            fp = int(((preds == 1) & (truth == 0)).sum())
            fn = int(((preds == 0) & (truth == 1)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        best_thresholds.append(best_thr)

    metrics = compute_binary_metrics(y_true, y_prob, best_thresholds)
    return best_thresholds, metrics


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_image_files(path: Path | str, recursive: bool = True) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if recursive:
        files = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in path.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def resolve_image_path(raw_path: str, csv_parent: Path | None = None, dataset_root: Path | None = None) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        if csv_parent is not None:
            candidates.append(csv_parent / path)
        if dataset_root is not None:
            candidates.append(dataset_root / path)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return candidates[-1].resolve() if candidates else path.resolve()


def load_checkpoint(checkpoint_path: Path | str, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model_from_checkpoint(checkpoint_path: Path | str, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model_name = checkpoint.get("model_name", "resnet18")
    image_size = int(checkpoint.get("image_size", 224))
    thresholds = checkpoint.get("thresholds", [0.5, 0.5])
    pretrained = False
    model = build_model(model_name=model_name, pretrained=pretrained, num_outputs=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "model_name": model_name,
        "image_size": image_size,
        "thresholds": thresholds,
        "class_names": checkpoint.get("class_names", CLASS_NAMES),
        "val_metrics": checkpoint.get("val_metrics", {}),
    }
    return model, meta


def annotate_image(
    image_bgr: np.ndarray,
    fire_prob: float,
    smoke_prob: float,
    fire_decision: bool,
    smoke_decision: bool,
) -> np.ndarray:
    frame = image_bgr.copy()
    labels: List[str] = []
    if fire_decision:
        labels.append("FIRE")
    if smoke_decision:
        labels.append("SMOKE")
    if not labels:
        labels.append("NORMAL")

    color = (0, 255, 0)
    if fire_decision or smoke_decision:
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Fire: {fire_prob:.2f} | Smoke: {smoke_prob:.2f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        " | ".join(labels),
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def create_video_writer(output_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    codecs = ["mp4v", "avc1", "MJPG"] if suffix == ".mp4" else ["XVID", "MJPG", "mp4v"]

    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError("Không tạo được VideoWriter. Hãy thử đổi output sang .avi hoặc cài FFmpeg/OpenCV đầy đủ.")
