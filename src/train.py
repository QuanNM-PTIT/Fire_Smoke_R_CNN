\
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    CLASS_NAMES,
    build_model,
    compute_binary_metrics,
    compute_pos_weight,
    ensure_dir,
    find_best_thresholds,
    get_device,
    get_transforms,
    save_json,
    seed_everything,
    sigmoid_np,
)
from dataset import FireSmokeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for fire and smoke classification.")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to val.csv")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional base folder for relative image paths")
    parser.add_argument("--output-dir", type=str, default="runs/exp", help="Folder to save checkpoints and logs")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v3_small"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience based on macro F1")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch instead of ImageNet weights")
    return parser.parse_args()


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    train_ds = FireSmokeDataset(
        csv_path=args.train_csv,
        transform=get_transforms(args.image_size, train=True),
        dataset_root=args.dataset_root,
    )
    val_ds = FireSmokeDataset(
        csv_path=args.val_csv,
        transform=get_transforms(args.image_size, train=False),
        dataset_root=args.dataset_root,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_df = pd.read_csv(args.train_csv)
    targets = train_df[["fire", "smoke"]].to_numpy(dtype=np.float32)
    return train_loader, val_loader, targets


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_probs = []
    losses = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, y_true, y_prob


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    device = get_device()
    print(f"[INFO] device = {device}")

    train_loader, val_loader, train_targets = build_loaders(args)
    pos_weight = compute_pos_weight(train_targets).to(device)
    print(f"[INFO] pos_weight = {pos_weight.tolist()}")

    model = build_model(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        num_outputs=2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history_path = output_dir / "history.csv"
    config_path = output_dir / "config.json"

    save_json(vars(args) | {"device": str(device)}, config_path)

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "fire_precision",
                "fire_recall",
                "fire_f1",
                "smoke_precision",
                "smoke_recall",
                "smoke_f1",
                "macro_f1",
                "exact_match_accuracy",
                "fire_threshold",
                "smoke_threshold",
                "epoch_seconds",
            ],
        )
        writer.writeheader()

    best_score = -1.0
    best_epoch = -1
    best_thresholds = [0.5, 0.5]
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for images, targets, _ in progress:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss, y_true, y_prob = evaluate(model, val_loader, criterion, device)
        thresholds, metrics = find_best_thresholds(y_true, y_prob)

        epoch_seconds = time.time() - epoch_start
        row: Dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "fire_precision": metrics["fire_precision"],
            "fire_recall": metrics["fire_recall"],
            "fire_f1": metrics["fire_f1"],
            "smoke_precision": metrics["smoke_precision"],
            "smoke_recall": metrics["smoke_recall"],
            "smoke_f1": metrics["smoke_f1"],
            "macro_f1": metrics["macro_f1"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "fire_threshold": thresholds[0],
            "smoke_threshold": thresholds[1],
            "epoch_seconds": epoch_seconds,
        }

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"fire_f1={metrics['fire_f1']:.4f} | smoke_f1={metrics['smoke_f1']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | thr={thresholds}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "thresholds": thresholds,
            "class_names": CLASS_NAMES,
            "val_metrics": metrics,
            "args": vars(args),
        }

        torch.save(checkpoint, output_dir / "last.pt")

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_score,
                    "thresholds": best_thresholds,
                    "val_metrics": metrics,
                },
                output_dir / "best_metrics.json",
            )
            print(f"[INFO] Saved new best model to {output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"[INFO] Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best macro_f1 = {best_score:.4f}"
            )
            break

    print("[DONE] Training finished.")
    print(f"[DONE] Best epoch = {best_epoch}")
    print(f"[DONE] Best thresholds = {best_thresholds}")


if __name__ == "__main__":
    main()
