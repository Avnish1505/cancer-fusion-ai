"""
v2 dataset + data plumbing shared by train_v2 and evaluate_v2.

Uses the SAME train/val/test split as v1 (src.dataset.stratified_split with
the same seed and CSV), on purpose: v1 and v2 must be compared on identical
test images, and the cached v1 logits in reports/ are in that order.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.dataset import DX_LABELS, LABEL_TO_IDX
from src.v2.metadata import MetadataEncoder


def resolve_image_paths(df: pd.DataFrame, image_dirs: list[str], ext: str = ".jpg") -> pd.DataFrame:
    """Adds an `image_path` column and drops rows whose file is missing,
    preserving row order (v1's HAM10000Dataset does the same skip)."""
    dirs = [Path(d) for d in image_dirs]
    paths = []
    for image_id in df["image_id"]:
        found = None
        for d in dirs:
            candidate = d / f"{image_id}{ext}"
            if candidate.exists():
                found = candidate
                break
        paths.append(found)
    out = df.copy()
    out["image_path"] = paths
    missing = out["image_path"].isna().sum()
    if missing:
        print(f"[v2.data] Warning: {missing} image(s) not found on disk, skipped.")
    out = out[out["image_path"].notna()].reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No images found under {image_dirs}")
    return out


class LesionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        encoder: MetadataEncoder | None,
        meta_dropout: float = 0.0,
        drop_all_metadata: bool = False,
    ):
        if "image_path" not in df.columns:
            raise ValueError("call resolve_image_paths() first")
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.encoder = encoder
        self.meta_dropout = meta_dropout
        self.drop_all_metadata = drop_all_metadata
        self.labels = np.array([LABEL_TO_IDX[d] for d in self.df["dx"]], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Unreadable image {row['image_path']}: {e}") from e
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.encoder is None:
            meta = torch.zeros(0)
        else:
            if self.drop_all_metadata:
                drop = (True, True, True)
            elif self.meta_dropout > 0:
                drop = tuple(random.random() < self.meta_dropout for _ in range(3))
            else:
                drop = (False, False, False)
            meta = torch.from_numpy(self.encoder.encode_row(row, drop))
        return image, meta, label


def class_weights(labels: np.ndarray, power: float, num_classes: int = len(DX_LABELS)) -> torch.Tensor:
    """w_c proportional to (1 / n_c) ** power, normalised to mean 1.
    power=1 is v1's full inverse frequency (over-weights df/vasc ~58x vs nv and
    buys minority recall with melanoma precision); 0.5 is the usual compromise."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(num_classes)
    w[present] = (1.0 / counts[present]) ** power
    w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)
