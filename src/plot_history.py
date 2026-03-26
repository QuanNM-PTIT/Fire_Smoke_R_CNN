\
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training history from history.csv")
    parser.add_argument("--history", type=str, required=True, help="Path to history.csv")
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    history_path = Path(args.history)
    df = pd.read_csv(history_path)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output) if args.output else history_path.with_name("loss_curve.png")
    fig.savefig(output_path, dpi=150)
    print(f"[DONE] Saved {output_path}")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["fire_f1"], label="fire_f1")
    plt.plot(df["epoch"], df["smoke_f1"], label="smoke_f1")
    plt.plot(df["epoch"], df["macro_f1"], label="macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1")
    plt.legend()
    plt.tight_layout()

    output_path_2 = history_path.with_name("f1_curve.png")
    fig.savefig(output_path_2, dpi=150)
    print(f"[DONE] Saved {output_path_2}")


if __name__ == "__main__":
    main()
