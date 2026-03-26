from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from common import create_video_writer, get_device
from faster_rcnn_utils import draw_detections, frame_to_tensor, load_checkpoint, postprocess_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster R-CNN on video and draw bounding boxes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt from train_faster_rcnn.py")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="predictions/faster_rcnn_video_out.mp4")
    parser.add_argument("--score-threshold", type=float, default=0.40)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--frame-stride", type=int, default=1, help="Run detector every N frames")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--display", action="store_true", help="Display frames while processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()
    model, meta = load_checkpoint(args.checkpoint, device=device)
    class_names = meta["class_names"]

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(output_path=output_path, fps=fps, size=(width, height))

    stride = max(1, int(args.frame_stride))
    frame_index = 0
    last_detections: list[dict] = []

    frame_rows: list[dict] = []
    det_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride == 0:
                tensor = frame_to_tensor(frame).to(device)
                output = model([tensor])[0]
                last_detections = postprocess_predictions(
                    output=output,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                )

            detections = last_detections
            annotated = draw_detections(
                image_bgr=frame,
                detections=detections,
                class_names=class_names,
                line_thickness=args.line_thickness,
            )
            cv2.putText(
                annotated,
                f"Frame: {frame_index} | Detections: {len(detections)}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            frame_rows.append(
                {
                    "frame": frame_index,
                    "detections": len(detections),
                }
            )

            for det_id, det in enumerate(detections):
                label = int(det["label"])
                class_name = class_names[label - 1] if 1 <= label <= len(class_names) else f"class_{label}"
                det_rows.append(
                    {
                        "frame": frame_index,
                        "detection_id": det_id,
                        "label": label,
                        "class_name": class_name,
                        "score": float(det["score"]),
                        "x1": float(det["x1"]),
                        "y1": float(det["y1"]),
                        "x2": float(det["x2"]),
                        "y2": float(det["y2"]),
                    }
                )

            if args.display:
                cv2.imshow("Faster R-CNN Fire/Smoke", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    frame_csv = output_path.with_name(output_path.stem + "_frame_summary.csv")
    with frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "detections"])
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    det_csv = output_path.with_name(output_path.stem + "_detections.csv")
    with det_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=["frame", "detection_id", "label", "class_name", "score", "x1", "y1", "x2", "y2"],
        )
        writer_csv.writeheader()
        writer_csv.writerows(det_rows)

    print(f"[DONE] Saved output video: {output_path}")
    print(f"[DONE] Saved frame summary: {frame_csv}")
    print(f"[DONE] Saved detections: {det_csv}")


if __name__ == "__main__":
    main()
