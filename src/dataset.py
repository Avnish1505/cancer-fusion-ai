"""
HAM10000 Dataset loader for Phase 1 (image-only baseline).

Design notes for Phase 2 (metadata fusion):
- The `HAM10000Dataset` class is extended to optionally handle tabular metadata
  alongside images. This is controlled by the `use_metadata` flag.
- When `use_metadata=True`, it processes three patient metadata fields:
  1. `age`: Fills missing values with the median, then standardizes the feature.
  2. `sex`: Fills missing values with 'unknown', then one-hot encodes.
  3. `localization`: Fills missing values with 'unknown', then one-hot encodes.
- A `StandardScaler` for age is fitted on the training data and then reused for
  validation and test sets to prevent data leakage.
Design notes:
- HAM10000 images are split across two folders (part_1, part_2) on Kaggle.
  We check both, and skip (with a warning) any row whose image genuinely
  can't be found, rather than crashing the whole run.
- Labels are the 'dx' column: akiec, bcc, bkl, df, mel, nv, vasc (7 classes).
- The dataset is heavily imbalanced (nv ~67% of samples), so we expose
  a helper to compute class weights for the loss function.
"""
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_TO_IDX = {label: i for i, label in enumerate(DX_LABELS)}

DX_FULL_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dirs: List[str],
        image_extension: str = ".jpg",
        transform=None,
        use_metadata: bool = False,        # NEW
        age_scaler: Optional[StandardScaler] = None,  # NEW — fit on train, reuse for val/test
        age_median: Optional[float] = None,           # NEW - fit on train, reuse for val/test
        metadata_columns: Optional[List[str]] = None, # NEW - pass train's columns for val/test
    ):
        """
        Args:
            dataframe: must contain columns 'image_id' and 'dx'
            image_dirs: list of directories to search for images (part_1, part_2)
            image_extension: file extension of images
            transform: torchvision transform pipeline
            use_metadata: if True, process and return tabular data alongside images
            age_scaler: if provided, use this scaler for age; otherwise, fit a new one
            age_median: if provided, use this value to fill missing ages; otherwise, compute it
            metadata_columns: if provided, align metadata to this exact column set
        """
        self.image_dirs = [Path(d) for d in image_dirs]
        self.image_extension = image_extension
        self.transform = transform
        self.use_metadata = use_metadata
        self.age_scaler = age_scaler
        self.age_median = age_median

        if self.use_metadata:
            df = dataframe.copy()

            # --- Age: numeric, handle NaN, then scale ---
            # For train set, compute median. For val/test, reuse train's median to prevent data leakage.
            if self.age_median is None:
                # This block runs for the training set
                self.age_median = df["age"].median()

            # Fill NaNs using the computed (or provided) median
            df["age"] = df["age"].fillna(self.age_median)

            if self.age_scaler is None:
                self.age_scaler = StandardScaler()
                df["age_scaled"] = self.age_scaler.fit_transform(df[["age"]])
            else:
                df["age_scaled"] = self.age_scaler.transform(df[["age"]])

            # --- Sex: one-hot (male, female, unknown) ---
            df["sex"] = df["sex"].fillna("unknown")
            sex_dummies = pd.get_dummies(df["sex"], prefix="sex", dtype=float)

            # --- Localization: one-hot (body site) ---
            df["localization"] = df["localization"].fillna("unknown")
            loc_dummies = pd.get_dummies(df["localization"], prefix="loc", dtype=float)

            metadata_df = pd.concat([df[["age_scaled"]], sex_dummies, loc_dummies], axis=1)

            if metadata_columns is not None:
                # val/test: align to train's exact column set
                self.metadata_df = metadata_df.reindex(columns=metadata_columns, fill_value=0)
                self.metadata_columns = metadata_columns
            else:
                # train: define the columns
                self.metadata_columns = ["age_scaled"] + list(sex_dummies.columns) + list(loc_dummies.columns)
                self.metadata_df = metadata_df.reindex(columns=self.metadata_columns, fill_value=0)

            dataframe = df  # so image_id/dx loop below still works

        # Pre-resolve image paths ONCE and drop rows whose image is missing.
        # This is safer than failing at __getitem__ time mid-training.
        resolved_rows = []
        missing_count = 0
        for i, row in dataframe.iterrows():
            img_path = self._resolve_image_path(row["image_id"])
            if img_path is None:
                missing_count += 1
                continue
            resolved_rows.append((img_path, row["dx"], i))


        if missing_count > 0:
            print(
                f"[HAM10000Dataset] Warning: {missing_count} image(s) referenced in "
                f"metadata could not be found on disk and were skipped."
            )

        if len(resolved_rows) == 0:
            raise RuntimeError(
                "No valid images found. Check that 'image_dirs' point to the correct "
                "HAM10000 image folders (HAM10000_images_part_1 / part_2)."
            )

        self.samples: List[Tuple[Path, str, int]] = resolved_rows

    def _resolve_image_path(self, image_id: str) -> Optional[Path]:
        filename = f"{image_id}{self.image_extension}"
        for directory in self.image_dirs:
            candidate = directory / filename
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, dx_label, orig_idx = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[HAM10000Dataset] Warning: corrupt image at {img_path} ({e}). Using blank fallback.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label_idx = LABEL_TO_IDX[dx_label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if self.use_metadata:
            meta_row = self.metadata_df.loc[orig_idx].values.astype("float32")
            meta_tensor = torch.tensor(meta_row, dtype=torch.float32)
            return image, meta_tensor, label_tensor

        return image, label_tensor




def load_metadata(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata CSV not found at '{csv_path}'. "
            f"On Kaggle, add the 'Skin Cancer MNIST: HAM10000' dataset to your notebook "
            f"and verify the path under /kaggle/input/."
        )

    df = pd.read_csv(path)

    required_cols = {"image_id", "dx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata CSV is missing required columns: {missing_cols}")

    unknown_labels = set(df["dx"].unique()) - set(DX_LABELS)
    if unknown_labels:
        raise ValueError(
            f"Found unexpected 'dx' labels not in the known 7 classes: {unknown_labels}. "
            f"Expected only: {DX_LABELS}"
        )

    # Drop rows with missing image_id or dx — better to skip than crash later
    before = len(df)
    df = df.dropna(subset=["image_id", "dx"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[load_metadata] Dropped {dropped} row(s) with missing image_id/dx.")

    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by lesion_id when available (to avoid the SAME lesion's images
    leaking across train/val/test — a common HAM10000 pitfall since some
    lesions have multiple photos), otherwise falls back to a plain stratified
    split on image_id.
    """
    total = train_split + val_split + test_split
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    group_col = "lesion_id" if "lesion_id" in df.columns else "image_id"

    # First split off the test set
    groups = df[group_col].unique()
    train_val_groups, test_groups = train_test_split(
        groups, test_size=test_split, random_state=random_seed
    )

    # Then split train_val into train and val
    relative_val_size = val_split / (train_split + val_split)
    train_groups, val_groups = train_test_split(
        train_val_groups, test_size=relative_val_size, random_state=random_seed
    )

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    print(
        f"[stratified_split] Train: {len(train_df)} | Val: {len(val_df)} | "
        f"Test: {len(test_df)} (split by '{group_col}' to avoid data leakage)"
    )

    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights, for use with nn.CrossEntropyLoss(weight=...)."""
    counts = df["dx"].value_counts()
    weights = []
    for label in DX_LABELS:
        count = counts.get(label, 0)
        if count == 0:
            print(f"[compute_class_weights] Warning: class '{label}' has 0 samples in this split.")
            weights.append(0.0)
        else:
            weights.append(1.0 / count)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so weights average to 1.0 (keeps loss scale stable)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(weights_tensor)
    return weights_tensor
