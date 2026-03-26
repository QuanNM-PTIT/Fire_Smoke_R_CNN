\
from __future__ import annotations

import argparse
import csv

import cv2
from PIL import Image
import torch

from common import (
    annotate_image,
    ensure_dir,
    get_device,
    get_transforms,
    list_image_files,
    load_model_from_checkpoint,
)
from gradcam_utils import GradCAM, combine_cams, find_target_layer, overlay_cam, pick_cam_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fire/smoke on one image or a folder of images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output-dir", type=str, default="predictions/images", help="Folder to save annotated images")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan folders for images")
    parser.add_argument("--fire-threshold", type=float, default=None)
    parser.add_argument("--smoke-threshold", type=float, default=None)
    parser.add_argument("--grad-cam", action="store_true", help="Overlay Grad-CAM heatmap")
    parser.add_argument("--cam-target", type=str, default="both", choices=["both", "fire", "smoke"])
    parser.add_argument("--cam-alpha", type=float, default=0.45, help="Heatmap blend factor")
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

    image_paths = list_image_files(args.input, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {args.input}")

    output_dir = ensure_dir(args.output_dir)
    rows = []
    grad_cam = None

    if args.grad_cam:
        target_layer = find_target_layer(model)
        grad_cam = GradCAM(model=model, target_layer=target_layer)
        print(f"[INFO] Grad-CAM target layer: {target_layer.__class__.__name__}")

    model.eval()
    for image_path in image_paths:
        pil_img = Image.open(image_path).convert("RGB")
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        fire_prob, smoke_prob = probs.tolist()

        fire_decision = fire_prob >= thresholds[0]
        smoke_decision = smoke_prob >= thresholds[1]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Bo qua anh loi: {image_path}")
            continue

        base_image = image_bgr
        if grad_cam is not None:
            classes_for_cam = pick_cam_classes(
                target=args.cam_target,
                fire_prob=float(fire_prob),
                smoke_prob=float(smoke_prob),
                min_prob=float(args.cam_min_prob),
            )
            cams = [grad_cam.generate(x, class_index=c) for c in classes_for_cam]
            weights = [float(probs[c]) for c in classes_for_cam]
            merged_cam = combine_cams(cams, weights=weights)
            base_image = overlay_cam(image_bgr, merged_cam, alpha=args.cam_alpha)

        annotated = annotate_image(
            image_bgr=base_image,
            fire_prob=fire_prob,
            smoke_prob=smoke_prob,
            fire_decision=fire_decision,
            smoke_decision=smoke_decision,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)

        row = {
            "image_path": str(image_path),
            "fire_prob": float(fire_prob),
            "smoke_prob": float(smoke_prob),
            "fire": int(fire_decision),
            "smoke": int(smoke_decision),
            "annotated_path": str(output_path),
        }
        rows.append(row)
        print(row)

    if grad_cam is not None:
        grad_cam.close()

    if not rows:
        raise RuntimeError("Khong co anh hop le de du doan.")

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved annotated images to: {output_dir}")
    print(f"[DONE] Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
