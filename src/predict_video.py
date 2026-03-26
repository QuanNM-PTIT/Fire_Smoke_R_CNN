from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from common import (
    annotate_image,
    create_video_writer,
    get_device,
    get_transforms,
    load_model_from_checkpoint,
)
from gradcam_utils import GradCAM, combine_cams, find_target_layer, overlay_cam, pick_cam_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fire/smoke on a video and save annotated output.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="predictions/output.mp4", help="Path to output video")
    parser.add_argument("--fire-threshold", type=float, default=None)
    parser.add_argument("--smoke-threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.80, help="EMA smoothing factor")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run model every N frames")
    parser.add_argument("--min-consecutive", type=int, default=3, help="Need this many consecutive frames to trigger alarm")
    parser.add_argument("--display", action="store_true", help="Display frames while processing")
    parser.add_argument("--grad-cam", action="store_true", help="Overlay Grad-CAM heatmap")
    parser.add_argument("--cam-target", type=str, default="both", choices=["both", "fire", "smoke"])
    parser.add_argument("--cam-alpha", type=float, default=0.35, help="Heatmap blend factor")
    parser.add_argument(
        "--cam-min-prob",
        type=float,
        default=0.20,
        help="For --cam-target both: include class in CAM only if prob >= this threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
    transform = get_transforms(image_size=int(meta["image_size"]), train=False)

    thresholds = list(meta["thresholds"])
    if args.fire_threshold is not None:
        thresholds[0] = args.fire_threshold
    if args.smoke_threshold is not None:
        thresholds[1] = args.smoke_threshold

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(output_path=output_path, fps=fps, size=(width, height))

    frame_index = 0
    ema_probs = np.zeros(2, dtype=np.float32)
    last_raw_probs = np.zeros(2, dtype=np.float32)
    fire_streak = 0
    smoke_streak = 0
    rows = []
    grad_cam = None
    last_cam = None

    if args.grad_cam:
        target_layer = find_target_layer(model)
        grad_cam = GradCAM(model=model, target_layer=target_layer)
        print(f"[INFO] Grad-CAM target layer: {target_layer.__class__.__name__}")

    model.eval()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % max(args.frame_stride, 1) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            x = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
            last_raw_probs = probs

            if grad_cam is not None:
                classes_for_cam = pick_cam_classes(
                    target=args.cam_target,
                    fire_prob=float(probs[0]),
                    smoke_prob=float(probs[1]),
                    min_prob=float(args.cam_min_prob),
                )
                cams = [grad_cam.generate(x, class_index=c) for c in classes_for_cam]
                weights = [float(probs[c]) for c in classes_for_cam]
                last_cam = combine_cams(cams, weights=weights)

        ema_probs = args.alpha * ema_probs + (1.0 - args.alpha) * last_raw_probs
        fire_prob, smoke_prob = ema_probs.tolist()

        if fire_prob >= thresholds[0]:
            fire_streak += 1
        else:
            fire_streak = 0

        if smoke_prob >= thresholds[1]:
            smoke_streak += 1
        else:
            smoke_streak = 0

        fire_decision = fire_streak >= args.min_consecutive
        smoke_decision = smoke_streak >= args.min_consecutive

        render_frame = frame
        if grad_cam is not None and last_cam is not None:
            render_frame = overlay_cam(render_frame, last_cam, alpha=args.cam_alpha)

        annotated = annotate_image(
            image_bgr=render_frame,
            fire_prob=fire_prob,
            smoke_prob=smoke_prob,
            fire_decision=fire_decision,
            smoke_decision=smoke_decision,
        )
        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(annotated)

        rows.append(
            {
                "frame": frame_index,
                "fire_prob": float(fire_prob),
                "smoke_prob": float(smoke_prob),
                "fire": int(fire_decision),
                "smoke": int(smoke_decision),
            }
        )

        if args.display:
            cv2.imshow("Fire / Smoke Demo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        frame_index += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()
    if grad_cam is not None:
        grad_cam.close()

    csv_path = output_path.with_name(output_path.stem + "_frame_scores.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame", "fire_prob", "smoke_prob", "fire", "smoke"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"[DONE] Saved video to: {output_path}")
    print(f"[DONE] Saved frame scores to: {csv_path}")


if __name__ == "__main__":
    main()
