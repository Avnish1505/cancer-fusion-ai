"""
Shared utilities: reproducibility, device selection, checkpointing.
"""
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets seeds across all RNG sources so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN — slightly slower but reproducible, important
    # when you're comparing Phase 1 vs Phase 2 results later.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Resolves the requested device, falling back to CPU with a warning
    instead of crashing if CUDA isn't actually available (e.g. running
    locally without a GPU vs. on Kaggle with one).
    """
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[get_device] Warning: CUDA requested but not available. Falling back to CPU. "
              "Training will be much slower — make sure GPU is enabled in Kaggle/Colab settings.")
        return torch.device("cpu")
    return torch.device(preferred)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_metric: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_metric": best_val_metric,
    }

    try:
        torch.save(checkpoint, checkpoint_path)
    except OSError as e:
        raise RuntimeError(
            f"Failed to save checkpoint to '{checkpoint_path}': {e}. "
            f"Check disk space and write permissions."
        )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    # Check if the checkpoint is a dictionary with 'model_state_dict'
    # or just the state_dict itself. This handles models saved with
    # torch.save(model.state_dict(), path) vs torch.save(checkpoint_dict, path)
    # and also helps with Git LFS pointer files being loaded by mistake.
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    try:
        # Use strict=False to be more robust to minor architecture changes
        # if a layer is added/removed, it won't crash immediately.
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint architecture doesn't match current model — "
            f"did the backbone/num_classes change since this checkpoint was saved? Details: {e}"
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class EarlyStopping:
    """Stops training when validation metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        improved = (
            current_score > self.best_score + self.min_delta
            if self.mode == "max"
            else current_score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
