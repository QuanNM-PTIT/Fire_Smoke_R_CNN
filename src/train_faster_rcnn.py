from __future__ import annotations

import argparse
import csv
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from common import ensure_dir, get_device, save_json, seed_everything
from faster_rcnn_data import YoloDetectionDataset, detection_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN from YOLO bbox labels.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--train-list", type=str, required=True, help="Path to train.txt")
    parser.add_argument("--val-list", type=str, required=True, help="Path to val.txt")
    parser.add_argument("--class-names", type=str, default="fire,smoke", help="Comma-separated class names")
    parser.add_argument("--output-dir", type=str, default="runs/faster_rcnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-step-size", type=int, default=8)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-random-hflip", action="store_true")
    return parser.parse_args()


def parse_class_names(raw: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("--class-names phai co it nhat 1 class")
    return names


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


def move_batch_to_device(images, targets, device: torch.device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    losses: List[float] = []

    progress = tqdm(loader, desc=f"Train {epoch}/{epochs}", ncols=110)
    for images, targets, _ in progress:
        images, targets = move_batch_to_device(images, targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return float(np.mean(losses)) if losses else 0.0


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.train()
    losses: List[float] = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = move_batch_to_device(images, targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    class_names = parse_class_names(args.class_names)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    print(f"[INFO] device = {device}")
    print(f"[INFO] classes = {class_names}")

    train_ds = YoloDetectionDataset(
        dataset_root=args.dataset_root,
        split_list=args.train_list,
        class_names=class_names,
        train=True,
        random_hflip=not args.no_random_hflip,
    )
    val_ds = YoloDetectionDataset(
        dataset_root=args.dataset_root,
        split_list=args.val_list,
        class_names=class_names,
        train=False,
        random_hflip=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    model = build_model(
        num_classes=len(class_names) + 1,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.lr_step_size)),
        gamma=float(args.lr_gamma),
    )

    save_json(vars(args) | {"device": str(device), "class_names": class_names}, output_dir / "config.json")

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "epoch_seconds"])
        writer.writeheader()

    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss = evaluate_loss(model, val_loader, device)

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "epoch_seconds": time.time() - epoch_start,
        }

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names) + 1,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                output_dir / "best_metrics.json",
            )
            print(
                f"[EPOCH {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={current_lr:.6f} | saved best"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[EPOCH {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={current_lr:.6f}"
            )

        scheduler.step()

        if epochs_without_improvement >= args.patience:
            print(
                f"[INFO] Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best_val_loss = {best_val:.4f}"
            )
            break

    print("[DONE] Training finished")
    print(f"[DONE] best.pt = {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
