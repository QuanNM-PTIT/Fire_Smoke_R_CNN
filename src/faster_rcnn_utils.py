from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision.transforms import functional as F


def build_model(num_classes: int, pretrained: bool, min_size: int, max_size: int) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    weights_backbone = None if pretrained else None

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("class_names", ["fire", "smoke"])
    num_classes = int(checkpoint.get("num_classes", len(class_names) + 1))

    args = checkpoint.get("args", {})
    min_size = int(args.get("min_size", 512))
    max_size = int(args.get("max_size", 1024))

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "class_names": class_names,
        "num_classes": num_classes,
        "min_size": min_size,
        "max_size": max_size,
        "args": args,
    }
    return model, meta


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return F.to_tensor(rgb)


def postprocess_predictions(
    output: Dict[str, torch.Tensor],
    score_threshold: float,
    nms_iou: float,
) -> List[Dict[str, float | int]]:
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    keep = scores >= float(score_threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if boxes.numel() == 0:
        return []

    keep_idx = nms(boxes, scores, float(nms_iou))
    boxes = boxes[keep_idx]
    labels = labels[keep_idx]
    scores = scores[keep_idx]

    detections: List[Dict[str, float | int]] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "label": int(labels[i].item()),
                "score": float(scores[i].item()),
            }
        )

    detections.sort(key=lambda d: float(d["score"]), reverse=True)
    return detections


def class_color(label: int) -> tuple[int, int, int]:
    palette = {
        1: (0, 0, 255),
        2: (0, 165, 255),
    }
    return palette.get(label, (80, 200, 120))


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, float | int]],
    class_names: List[str],
    line_thickness: int = 2,
) -> np.ndarray:
    frame = image_bgr.copy()
    thickness = max(1, int(line_thickness))

    for det in detections:
        x1 = int(round(float(det["x1"])))
        y1 = int(round(float(det["y1"])))
        x2 = int(round(float(det["x2"])))
        y2 = int(round(float(det["y2"])))
        label = int(det["label"])
        score = float(det["score"])

        class_idx = label - 1
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"class_{label}"
        text = f"{class_name} {score:.2f}"
        color = class_color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, thickness)
        text_y = max(y1 - 8, text_h + baseline + 2)
        cv2.rectangle(
            frame,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 8, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return frame
