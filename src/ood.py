"""
Out-of-distribution (OOD) guard — analysis only. Does not change app.py.

Three scores, all read off the trained fusion model with no retraining:
- MSP: max softmax probability at T=2.1235 (app.py's serving temperature).
- Energy: negative logsumexp of RAW logits (Liu et al. 2020), T=1 by definition
  of the score itself — independent of the calibration temperature.
- Mahalanobis: class-conditional Mahalanobis distance (Lee et al. 2018) on the
  image encoder's 2048-d penultimate feature (image branch only, before
  metadata fusion — OOD-ness is a property of the photo, not the form fields).
  Means/shared covariance are fit on TRAINING split image features.

OOD images carry no patient metadata. For the two logit-based scores (MSP,
Energy) we run them with a single FIXED metadata vector — median age, most
common sex, most common site, all from the training split — repeated for
every OOD image. This is an approximation: it is NOT what a real user would
have entered, and it is called out explicitly wherever those scores are used.
The Mahalanobis score never touches metadata.

Usage:
    python -m src.ood \
        --ham-images-dir-part1 /path/to/HAM10000_images_part_1 \
        --ham-images-dir-part2 /path/to/HAM10000_images_part_2 \
        --coco-images-dir /path/to/coco_sample \
        --pad-ufes-images-dir /path/to/pad_ufes_sample \
        --synthetic-images-dir /path/to/synthetic
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset

from src.config import ConfigError, load_config
from src.dataset import DX_LABELS, HAM10000Dataset, load_metadata, stratified_split
from src.evaluate_isic2018 import fmt_ci
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint

CURRENT_TEMPERATURE = 2.1235  # matches app.py; scores are read-only, never fit here
N_BOOTSTRAP_DEFAULT = 1000
VAL_ACCEPT_RATE = 0.95  # thresholds chosen so 95% of internal val is accepted
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]
MALIGNANT_IDX = [DX_LABELS.index(c) for c in MALIGNANT_CLASSES]


# ---------------------------------------------------------------------------
# Model helpers: one forward pass -> both the 2048-d image feature and logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_features_and_logits(model, images: torch.Tensor, metadata: torch.Tensor | None):
    image_features = model.forward_features(images)
    if model.use_metadata:
        meta_features = model.metadata_processor(metadata)
        fused = torch.cat((image_features, meta_features), dim=1)
        logits = model.classifier_head(fused)
    else:
        logits = model.classifier_head(image_features)
    return image_features, logits


def encode_fixed_metadata(age_scaler, age_median: float, metadata_columns: list[str], sex: str, localization: str) -> torch.Tensor:
    age_scaled = float(age_scaler.transform([[age_median]])[0][0])
    row = {col: 0.0 for col in metadata_columns}
    row["age_scaled"] = age_scaled
    sex_col = f"sex_{sex}"
    if sex_col in row:
        row[sex_col] = 1.0
    loc_col = f"loc_{localization}"
    if loc_col in row:
        row[loc_col] = 1.0
    vector = [row[c] for c in metadata_columns]
    return torch.tensor(vector, dtype=torch.float32)


class RawImageDataset(Dataset):
    """Loads a flat directory of arbitrary images — no labels, no metadata CSV."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        image = Image.open(p).convert("RGB")
        return self.transform(image), p.name


@torch.no_grad()
def run_image_folder(model, paths: list[Path], transform, device, metadata_vector: torch.Tensor | None,
                      batch_size: int = 32, num_workers: int = 2):
    ds = RawImageDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_features, all_logits, all_names = [], [], []
    for images, names in loader:
        images = images.to(device)
        meta = None
        if metadata_vector is not None:
            meta = metadata_vector.unsqueeze(0).repeat(images.shape[0], 1).to(device)
        feats, logits = forward_features_and_logits(model, images, meta)
        all_features.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_names.extend(names)
    return torch.cat(all_features), torch.cat(all_logits), all_names


@torch.no_grad()
def run_ham_dataset(model, dataset, device, use_metadata: bool, batch_size: int, num_workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_features, all_logits, all_labels = [], [], []
    for batch in loader:
        if use_metadata:
            images, metas, labels = batch
            images, metas = images.to(device), metas.to(device)
        else:
            images, labels = batch
            images, metas = images.to(device), None
        feats, logits = forward_features_and_logits(model, images, metas)
        all_features.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_logits), torch.cat(all_labels)


def cached(cache_dir: Path, filename: str, compute_fn, force_recompute: bool = False):
    path = cache_dir / filename
    if not force_recompute and path.exists():
        print(f"[ood] Using cached {path}")
        return torch.load(path)
    result = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"[ood] Cached {path}")
    return result


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def msp_score(logits: torch.Tensor, T: float = CURRENT_TEMPERATURE) -> np.ndarray:
    return F.softmax(logits / T, dim=1).max(dim=1).values.numpy()


def energy_score(logits: torch.Tensor) -> np.ndarray:
    """Negative logsumexp of RAW logits — higher means more OOD-like."""
    return (-torch.logsumexp(logits, dim=1)).numpy()


def fit_mahalanobis(train_features: torch.Tensor, train_labels: torch.Tensor, n_classes: int):
    """
    Per-class means + one shared (pooled within-class) covariance, Lee et al. 2018.

    The covariance is estimated with Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004),
    not a plain inverse. With ~6,979 training images and 2048-d features (n/d
    ratio ~3.4), the raw pooled sample covariance is technically invertible
    (no zero/negative eigenvalues) but very poorly conditioned — condition
    number ~1.4e5 in practice, versus ~2.5e4 after Ledoit-Wolf shrinkage
    (shrinkage intensity ~1%, chosen automatically from the data, not a fixed
    magic number). This is exactly the small-n/d regime shrinkage estimators
    are designed for (Marchenko-Pastur eigenvalue distortion). Empirically the
    shrinkage barely moves AUROC/FPR95/malignant-rejection on this project's
    OOD sets (all within bootstrap noise) — it's adopted for principled
    conditioning, not because it changes the verdict.
    """
    feats = train_features.numpy().astype(np.float64)
    labels = train_labels.numpy()
    d = feats.shape[1]
    means = np.zeros((n_classes, d))
    centered_chunks = []
    for k in range(n_classes):
        mask = labels == k
        if mask.sum() == 0:
            continue
        class_feats = feats[mask]
        mu = class_feats.mean(axis=0)
        means[k] = mu
        centered_chunks.append(class_feats - mu)
    centered_all = np.vstack(centered_chunks)
    lw = LedoitWolf().fit(centered_all)
    precision = np.linalg.inv(lw.covariance_)
    return means, precision


def mahalanobis_score(features: torch.Tensor, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    feats = features.numpy().astype(np.float64)
    n_classes = means.shape[0]
    dists = np.empty((feats.shape[0], n_classes))
    for k in range(n_classes):
        diff = feats - means[k]
        dists[:, k] = np.einsum("ij,jk,ik->i", diff, precision, diff)
    return dists.min(axis=1)


# ---------------------------------------------------------------------------
# Threshold selection (internal val only) + evaluation
# ---------------------------------------------------------------------------

def accept_threshold_low_is_ood(val_scores: np.ndarray, accept_rate: float) -> float:
    """For scores where LOWER = more in-distribution (energy, mahalanobis):
    accept if score <= threshold. threshold = accept_rate-th percentile of val."""
    return float(np.quantile(val_scores, accept_rate))


def accept_threshold_high_is_ood(val_scores: np.ndarray, accept_rate: float) -> float:
    """For MSP, where HIGHER = more in-distribution: accept if score >= threshold."""
    return float(np.quantile(val_scores, 1 - accept_rate))


def bootstrap_ci_two_sample(n_a: int, n_b: int, metric_fn, n_boot: int = N_BOOTSTRAP_DEFAULT, seed: int = 42):
    """Independent two-sample percentile bootstrap: metric_fn(idx_a, idx_b) -> float."""
    point = metric_fn(np.arange(n_a), np.arange(n_b))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        val = metric_fn(idx_a, idx_b)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            boots.append(val)
    if len(boots) < n_boot // 2:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def fpr_at_tpr(y_true_ood: np.ndarray, ood_ness_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = skmetrics.roc_curve(y_true_ood, ood_ness_score)
    return float(np.interp(target_tpr, tpr, fpr))


def auroc_and_fpr95(id_score: np.ndarray, ood_score: np.ndarray, idx_id: np.ndarray, idx_ood: np.ndarray):
    y = np.concatenate([np.zeros(len(idx_id)), np.ones(len(idx_ood))])
    s = np.concatenate([id_score[idx_id], ood_score[idx_ood]])
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(skmetrics.roc_auc_score(y, s))


def fpr95_metric(id_score: np.ndarray, ood_score: np.ndarray, idx_id: np.ndarray, idx_ood: np.ndarray):
    y = np.concatenate([np.zeros(len(idx_id)), np.ones(len(idx_ood))])
    s = np.concatenate([id_score[idx_id], ood_score[idx_ood]])
    if len(np.unique(y)) != 2:
        return float("nan")
    return fpr_at_tpr(y, s)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_score_report(score_name: str, ood_ness_id: dict, ood_ness_ood_sets: dict, n_boot: int, seed: int):
    """ood_ness_id: {'internal_test': arr, 'external_isic2018': arr} (pooled for AUROC/FPR95).
    ood_ness_ood_sets: {ood_name: arr}."""
    print(f"\n{'=' * 78}\n{score_name} — OOD-ness score (higher = more OOD-like)\n{'=' * 78}")
    id_pool = np.concatenate(list(ood_ness_id.values()))
    for ood_name, ood_arr in ood_ness_ood_sets.items():
        auroc_pt, auroc_lo, auroc_hi = bootstrap_ci_two_sample(
            len(id_pool), len(ood_arr),
            lambda idx_id, idx_ood: auroc_and_fpr95(id_pool, ood_arr, idx_id, idx_ood),
            n_boot, seed,
        )
        fpr_pt, fpr_lo, fpr_hi = bootstrap_ci_two_sample(
            len(id_pool), len(ood_arr),
            lambda idx_id, idx_ood: fpr95_metric(id_pool, ood_arr, idx_id, idx_ood),
            n_boot, seed,
        )
        print(f"  vs {ood_name:14s} (n_ood={len(ood_arr):4d}, ID pool n={len(id_pool)}): "
              f"AUROC={fmt_ci(auroc_pt, auroc_lo, auroc_hi)}   FPR@95%TPR={fmt_ci(fpr_pt, fpr_lo, fpr_hi)}")


# ---------------------------------------------------------------------------
# Example grid figures (COCO + synthetic only — never PAD-UFES patient images)
# ---------------------------------------------------------------------------

def save_example_grid(paths: list[Path], names: list[str], msp: np.ndarray, energy: np.ndarray,
                       maha: np.ndarray, preds: np.ndarray, out_dir: Path, set_name: str, n_examples: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    name_to_idx = {n: i for i, n in enumerate(names)}
    chosen = rng.choice(len(paths), size=min(n_examples, len(paths)), replace=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    ncols = 4
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, i in zip(axes, chosen):
        p = paths[i]
        idx = name_to_idx[p.name]
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"pred={DX_LABELS[preds[idx]]}  MSP={msp[idx]:.3f}\nEnergy={energy[idx]:.2f}  Maha={maha[idx]:.1f}",
            fontsize=9,
        )
        manifest.append({
            "file": p.name, "predicted_class": DX_LABELS[preds[idx]],
            "msp": float(msp[idx]), "energy": float(energy[idx]), "mahalanobis": float(maha[idx]),
        })
        (out_dir / p.name).write_bytes(p.read_bytes())
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle(f"{set_name} — example images, unguarded model output")
    fig.tight_layout()
    fig.savefig(out_dir / f"_{set_name}_grid.png", dpi=130)
    plt.close(fig)
    (out_dir / f"_{set_name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ood] Saved {len(chosen)} example images + grid to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OOD guard analysis: MSP, Energy, Mahalanobis scores.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--ham-metadata-csv", type=str, default="data/HAM10000_metadata.csv")
    parser.add_argument("--ham-images-dir-part1", type=str, required=True)
    parser.add_argument("--ham-images-dir-part2", type=str, required=True)
    parser.add_argument("--coco-images-dir", type=str, required=True)
    parser.add_argument("--pad-ufes-images-dir", type=str, required=True)
    parser.add_argument("--synthetic-images-dir", type=str, required=True)
    parser.add_argument("--isic-images-dir", type=str, required=True, help="For internal-test/external feature extraction consistency (external ISIC test images).")
    parser.add_argument("--cache-dir", type=str, default="reports/ood/cache")
    parser.add_argument("--output-dir", type=str, default="reports/ood")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples").mkdir(parents=True, exist_ok=True)

    device = get_device(config["train"]["device"])
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[ood] CUDA unavailable; using MPS instead of CPU.")

    df = load_metadata(args.ham_metadata_csv)
    train_df, val_df, test_df = stratified_split(
        df, config["data"]["train_split"], config["data"]["val_split"], config["data"]["test_split"],
        config["data"]["random_seed"],
    )
    ham_image_dirs = [args.ham_images_dir_part1, args.ham_images_dir_part2]
    image_size = config["train"]["image_size"]
    transform = get_eval_transforms(image_size)

    train_dataset = HAM10000Dataset(train_df, ham_image_dirs, ".jpg", transform=transform, use_metadata=True)
    age_scaler, age_median, metadata_columns = train_dataset.age_scaler, train_dataset.age_median, train_dataset.metadata_columns
    config["model"]["metadata_dim"] = len(metadata_columns)

    fixed_sex = train_df["sex"].mode(dropna=True)[0]
    fixed_localization = train_df["localization"].mode(dropna=True)[0]
    print(f"[ood] Fixed OOD metadata: age={age_median} (train median), sex={fixed_sex!r} (train mode), "
          f"localization={fixed_localization!r} (train mode). This is an APPROXIMATION for logit-based "
          f"scores (MSP, Energy) on images that have no real patient metadata.")
    fixed_meta_vector = encode_fixed_metadata(age_scaler, age_median, metadata_columns, fixed_sex, fixed_localization)

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    batch_size, num_workers = config["train"]["batch_size"], config["train"]["num_workers"]

    # ---- Training split features (fit Mahalanobis on this) ----
    train_result = cached(
        cache_dir, "train.pt",
        lambda: run_ham_dataset(model, train_dataset, device, True, batch_size, num_workers),
        args.force_recompute,
    )
    train_features, _, train_labels = train_result
    means, precision = fit_mahalanobis(train_features, train_labels, len(DX_LABELS))
    print(f"[ood] Fit Mahalanobis means/covariance on {len(train_labels)} training images "
          f"({train_features.shape[1]}-d encoder features).")

    # ---- Internal val (threshold selection only) ----
    val_dataset = HAM10000Dataset(val_df, ham_image_dirs, ".jpg", transform=transform, use_metadata=True,
                                   age_scaler=age_scaler, age_median=age_median, metadata_columns=metadata_columns)
    val_features, val_logits, val_labels_t = cached(
        cache_dir, "val.pt",
        lambda: run_ham_dataset(model, val_dataset, device, True, batch_size, num_workers),
        args.force_recompute,
    )

    # ---- Internal test (image files needed for features; logits already cached elsewhere but recomputed here for consistency with features) ----
    test_dataset = HAM10000Dataset(test_df, ham_image_dirs, ".jpg", transform=transform, use_metadata=True,
                                    age_scaler=age_scaler, age_median=age_median, metadata_columns=metadata_columns)
    test_features, test_logits, test_labels_t = cached(
        cache_dir, "internal_test.pt",
        lambda: run_ham_dataset(model, test_dataset, device, True, batch_size, num_workers),
        args.force_recompute,
    )

    # ---- External ISIC 2018 test ----
    isic_gt_path = Path(args.isic_images_dir).parent / "ISIC2018_Task3_Test_GroundTruth.tab"
    if not isic_gt_path.exists():
        # fall back: look for it alongside the images dir under the same name pattern used in evaluate_isic2018
        candidates = list(Path(args.isic_images_dir).parent.glob("*GroundTruth*.tab"))
        if candidates:
            isic_gt_path = candidates[0]
    isic_df = pd.read_csv(isic_gt_path, sep="\t", na_values=["", "NA"])
    isic_dataset = HAM10000Dataset(isic_df, [args.isic_images_dir], ".jpg", transform=transform, use_metadata=True,
                                    age_scaler=age_scaler, age_median=age_median, metadata_columns=metadata_columns)
    ext_features, ext_logits, ext_labels_t = cached(
        cache_dir, "external_isic2018.pt",
        lambda: run_ham_dataset(model, isic_dataset, device, True, batch_size, num_workers),
        args.force_recompute,
    )

    # ---- OOD sets ----
    def load_paths(d):
        exts = {".jpg", ".jpeg", ".png"}
        return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in exts)

    coco_paths = load_paths(args.coco_images_dir)
    pad_paths = load_paths(args.pad_ufes_images_dir)
    synth_paths = load_paths(args.synthetic_images_dir)
    print(f"[ood] OOD set sizes: COCO={len(coco_paths)}  PAD-UFES-20={len(pad_paths)}  synthetic={len(synth_paths)}")

    coco_features, coco_logits, coco_names = cached(
        cache_dir, "coco.pt",
        lambda: run_image_folder(model, coco_paths, transform, device, fixed_meta_vector, batch_size, num_workers),
        args.force_recompute,
    )
    pad_features, pad_logits, pad_names = cached(
        cache_dir, "pad_ufes.pt",
        lambda: run_image_folder(model, pad_paths, transform, device, fixed_meta_vector, batch_size, num_workers),
        args.force_recompute,
    )
    synth_features, synth_logits, synth_names = cached(
        cache_dir, "synthetic.pt",
        lambda: run_image_folder(model, synth_paths, transform, device, fixed_meta_vector, batch_size, num_workers),
        args.force_recompute,
    )

    # ---- Scores ----
    def compute_scores(features, logits):
        return {
            "MSP": msp_score(logits),
            "Energy": energy_score(logits),
            "Mahalanobis": mahalanobis_score(features, means, precision),
        }

    val_scores = compute_scores(val_features, val_logits)
    test_scores = compute_scores(test_features, test_logits)
    ext_scores = compute_scores(ext_features, ext_logits)
    coco_scores = compute_scores(coco_features, coco_logits)
    pad_scores = compute_scores(pad_features, pad_logits)
    synth_scores = compute_scores(synth_features, synth_logits)

    val_labels = val_labels_t.numpy()
    test_labels = test_labels_t.numpy()
    ext_labels = ext_labels_t.numpy()

    print("\n" + "=" * 78)
    print(f"THRESHOLDS chosen on internal val only, targeting {VAL_ACCEPT_RATE:.0%} in-distribution acceptance")
    print("=" * 78)
    thresholds = {}
    for name in ("MSP", "Energy", "Mahalanobis"):
        if name == "MSP":
            t = accept_threshold_high_is_ood(val_scores[name], VAL_ACCEPT_RATE)
            accepted = (val_scores[name] >= t).mean()
        else:
            t = accept_threshold_low_is_ood(val_scores[name], VAL_ACCEPT_RATE)
            accepted = (val_scores[name] <= t).mean()
        thresholds[name] = t
        print(f"  {name:12s} threshold={t:.4f}  (val acceptance at this threshold: {accepted:.4f})")

    def ood_ness(name, scores):
        return (1 - scores) if name == "MSP" else scores

    id_scores_by_set = {"internal_test": test_scores, "external_isic2018": ext_scores}
    ood_scores_by_set = {"COCO_far_ood": coco_scores, "PAD_UFES_20_near_ood": pad_scores, "synthetic_trivial": synth_scores}

    print("\n" + "=" * 78)
    print("AUROC / FPR@95%TPR — ID pool = internal_test + external_isic2018 (pooled), per OOD set")
    print("=" * 78)
    for score_name in ("MSP", "Energy", "Mahalanobis"):
        ood_ness_id = {k: ood_ness(score_name, v[score_name]) for k, v in id_scores_by_set.items()}
        ood_ness_ood = {k: ood_ness(score_name, v[score_name]) for k, v in ood_scores_by_set.items()}
        print_score_report(score_name, ood_ness_id, ood_ness_ood, args.n_bootstrap, args.seed)

    print("\n" + "=" * 78)
    print("FALSE REJECTION on in-distribution data, at the val-derived thresholds")
    print("=" * 78)
    for score_name in ("MSP", "Energy", "Mahalanobis"):
        t = thresholds[score_name]
        for set_name, labels, scores_dict, mal_idx in (
            ("internal_test", test_labels, test_scores, MALIGNANT_IDX),
            ("external_isic2018", ext_labels, ext_scores, MALIGNANT_IDX),
        ):
            s = scores_dict[score_name]
            rejected = (s < t) if score_name == "MSP" else (s > t)
            frac_rejected = float(rejected.mean())
            mal_mask = np.isin(labels, mal_idx)
            n_mal = int(mal_mask.sum())
            n_mal_rejected = int(rejected[mal_mask].sum())
            print(f"  {score_name:12s} {set_name:20s} wrongly rejected: {frac_rejected:.4f} "
                  f"({int(rejected.sum())}/{len(labels)})   malignant cases rejected: {n_mal_rejected}/{n_mal}")

    print("\n" + "=" * 78)
    print("UNGUARDED model behavior on OOD images (no guard applied — what does it tell the user today)")
    print("=" * 78)
    for set_name, paths, features, logits in (
        ("COCO (far-OOD)", coco_paths, coco_features, coco_logits),
        ("PAD-UFES-20 (near-OOD)", pad_paths, pad_features, pad_logits),
        ("synthetic (trivial)", synth_paths, synth_features, synth_logits),
    ):
        msp = msp_score(logits)
        preds = logits.argmax(dim=1).numpy()
        counts = np.bincount(preds, minlength=len(DX_LABELS))
        dist = {DX_LABELS[i]: int(counts[i]) for i in range(len(DX_LABELS)) if counts[i] > 0}
        frac_confident = float((msp > 0.5).mean())
        print(f"  {set_name:24s} n={len(paths):4d}  predicted-class counts: {dist}")
        print(f"      fraction with MSP > 0.5 confidence: {frac_confident:.4f}  (mean MSP={msp.mean():.4f})")

    # ---- Figures ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, score_name in zip(axes, ("MSP", "Energy", "Mahalanobis")):
        data = [val_scores[score_name], test_scores[score_name], ext_scores[score_name],
                coco_scores[score_name], pad_scores[score_name], synth_scores[score_name]]
        labels_x = ["val", "int.test", "ext.ISIC", "COCO", "PAD-UFES", "synth"]
        ax.boxplot(data, tick_labels=labels_x, showfliers=False)
        ax.set_title(score_name)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("OOD score distributions: ID sets vs. OOD sets")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"\n[ood] Saved {output_dir / 'score_distributions.png'}")

    # ---- Example images (COCO + synthetic only) ----
    coco_msp, coco_energy, coco_maha = coco_scores["MSP"], coco_scores["Energy"], coco_scores["Mahalanobis"]
    coco_preds = coco_logits.argmax(dim=1).numpy()
    save_example_grid(coco_paths, coco_names, coco_msp, coco_energy, coco_maha, coco_preds,
                       output_dir / "examples" / "coco", "coco", n_examples=8, seed=args.seed)

    synth_msp, synth_energy, synth_maha = synth_scores["MSP"], synth_scores["Energy"], synth_scores["Mahalanobis"]
    synth_preds = synth_logits.argmax(dim=1).numpy()
    save_example_grid(synth_paths, synth_names, synth_msp, synth_energy, synth_maha, synth_preds,
                       output_dir / "examples" / "synthetic", "synthetic", n_examples=8, seed=args.seed)
    print("[ood] PAD-UFES-20 examples intentionally NOT saved (real patient images).")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"[FILE ERROR] {e}")
        raise SystemExit(1)
