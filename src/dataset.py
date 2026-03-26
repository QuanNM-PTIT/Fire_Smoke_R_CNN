\
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import resolve_image_path


class FireSmokeDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.df = pd.read_csv(self.csv_path)

        expected_columns = {"path", "fire", "smoke"}
        if not expected_columns.issubset(set(self.df.columns)):
            raise ValueError(
                f"{self.csv_path} phải có các cột: {sorted(expected_columns)}. "
                f"Các cột hiện tại: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[index]
        image_path = resolve_image_path(
            str(row["path"]),
            csv_parent=self.csv_path.parent,
            dataset_root=self.dataset_root,
        )
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor([float(row["fire"]), float(row["smoke"])], dtype=torch.float32)
        return image, target, str(image_path)
