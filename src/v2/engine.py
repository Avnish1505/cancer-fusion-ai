"""Training/eval plumbing for v2: EMA, schedule, TTA inference, metrics, config overrides."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn

from src.dataset import DX_LABELS
from src.v2.transforms import dihedral_views

MEL = DX_LABELS.index("mel")
MALIGNANT_IDX = [DX_LABELS.index(c) for c in ("mel", "bcc", "akiec")]


# ---------------------------------------------------------------- config
def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """--set train.use_metadata=false --set model.backbone=convnext_tiny.fb_in22k"""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b=value, got '{item}'")
        key, raw = item.split("=", 1)
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config


# ---------------------------------------------------------------- EMA
class ModelEma:
    """Exponential moving average of weights, with the usual warmup so the
    average isn't dominated by the random-init head in the first steps."""

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def warmup_cosine(total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device, tta_views: int = 1,
                   amp: bool = False, return_features: bool = False):
    """Returns (logits, labels[, image_features]) for the whole loader.
    With tta_views > 1, logits are averaged over dihedral views."""
    model.eval()
    all_logits, all_labels, all_feats = [], [], []
    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True) if meta.numel() else None
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            views = dihedral_views(images, tta_views) if tta_views > 1 else [images]
            logit_sum = 0
            for i, v in enumerate(views):
                feats = model.forward_features(v)
                if return_features and i == 0:
                    all_feats.append(feats.float().cpu())
                logit_sum = logit_sum + model.head(feats, meta).float()
        all_logits.append((logit_sum / len(views)).cpu())
        all_labels.append(labels)
    out = (torch.cat(all_logits), torch.cat(all_labels))
    return out + (torch.cat(all_feats),) if return_features else out


# ---------------------------------------------------------------- metrics
def summarize(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    lg = logits.numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    pred = lg.argmax(1)
    probs = torch.softmax(torch.as_tensor(lg), 1).numpy()
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mel_recall": recall_score(y, pred, labels=[MEL], average=None, zero_division=0)[0],
        "mel_precision": precision_score(y, pred, labels=[MEL], average=None, zero_division=0)[0],
    }
    # AUCs are threshold-free: they measure ranking quality, which post-hoc
    # decision tweaks cannot change. That makes them the honest "did the
    # representation get better" number.
    if len(np.unique(y == MEL)) == 2:
        out["mel_auc"] = roc_auc_score(y == MEL, probs[:, MEL])
    mal = np.isin(y, MALIGNANT_IDX)
    if len(np.unique(mal)) == 2:
        out["malignant_auc"] = roc_auc_score(mal, probs[:, MALIGNANT_IDX].sum(1))
    return {k: float(v) for k, v in out.items()}


def paired_bootstrap(metric_fn, y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """95% CI of metric(b) - metric(a) over bootstrap resamples of the SAME
    images. Paired, so it answers 'is v2 better on these images' rather than
    comparing two independently noisy numbers."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            deltas.append(metric_fn(y[idx], b[idx]) - metric_fn(y[idx], a[idx]))
        except ValueError:  # e.g. AUC on a resample with one class
            continue
    return float(metric_fn(y, b) - metric_fn(y, a)), np.percentile(deltas, [2.5, 97.5]).tolist()
