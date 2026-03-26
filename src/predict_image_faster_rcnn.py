from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from common import ensure_dir, get_device, list_image_files
from faster_rcnn_utils import draw_detections, frame_to_tensor, load_checkpoint, postprocess_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster R-CNN on image(s) and draw bounding boxes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt from train_faster_rcnn.py")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output-dir", type=str, default="predictions/faster_rcnn_images")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.40)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--line-thickness", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()
    model, meta = load_checkpoint(args.checkpoint, device=device)
    class_names = meta["class_names"]

    image_paths = list_image_files(args.input, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"Khong tim thay anh nao trong {args.input}")

    output_dir = ensure_dir(args.output_dir)

    rows: list[dict] = []
    summary_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[WARN] Bo qua anh loi: {image_path}")
                continue

            tensor = frame_to_tensor(image_bgr).to(device)
            output = model([tensor])[0]
            detections = postprocess_predictions(
                output=output,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
            )

            annotated = draw_detections(
                image_bgr=image_bgr,
                detections=detections,
                class_names=class_names,
                line_thickness=args.line_thickness,
            )

            out_path = output_dir / image_path.name
            cv2.imwrite(str(out_path), annotated)

            summary_rows.append(
                {
                    "image_path": str(image_path),
                    "detections": len(detections),
                    "annotated_path": str(out_path),
                }
            )

            for det_id, det in enumerate(detections):
                label = int(det["label"])
                class_name = class_names[label - 1] if 1 <= label <= len(class_names) else f"class_{label}"
                rows.append(
                    {
                        "image_path": str(image_path),
                        "detection_id": det_id,
                        "label": label,
                        "class_name": class_name,
                        "score": float(det["score"]),
                        "x1": float(det["x1"]),
                        "y1": float(det["y1"]),
                        "x2": float(det["x2"]),
                        "y2": float(det["y2"]),
                        "annotated_path": str(out_path),
                    }
                )

            print(f"[INFO] {image_path.name}: {len(detections)} detections")

    summary_csv = output_dir / "predictions_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "detections", "annotated_path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    detections_csv = output_dir / "predictions_detections.csv"
    with detections_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "detection_id", "label", "class_name", "score", "x1", "y1", "x2", "y2", "annotated_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved annotated images to: {output_dir}")
    print(f"[DONE] Saved summary CSV to: {summary_csv}")
    print(f"[DONE] Saved detections CSV to: {detections_csv}")


if __name__ == "__main__":
    main()
