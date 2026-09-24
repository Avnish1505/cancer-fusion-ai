"""
Post-hoc temperature scaling for the trained fusion model, following
Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks"
(ICML 2017).

Temperature scaling fits a single scalar T > 0 that divides the logits
before softmax. It is fit on the VALIDATION set (never on test — val was
already used for early stopping during training, but it was never used to
fit T, so it's still valid for that), by minimizing negative log-likelihood.
Dividing all logits by a positive scalar is monotonic, so it cannot change
argmax / accuracy / F1 — it only reshapes the confidence distribution.

This script does NOT touch train.py, evaluate.py, or app.py, and does NOT
change the hand-picked TEMPERATURE constant currently serving in app.py.
It only reports numbers so that value can be revisited deliberately.

Usage:
    python -m src.calibrate --config configs/config.yaml --checkpoint models/best_model.pt \
        --metadata-csv data/HAM10000_metadata.csv \
        --images-dir-part1 /path/to/HAM10000_images_part_1 \
        --images-dir-part2 /path/to/HAM10000_images_part_2
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import ConfigError, load_config
from src.dataset import DX_LABELS, HAM10000Dataset, load_metadata, stratified_split
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint

HAND_PICKED_T = 2.5  # the value currently hardcoded in app.py — kept only for comparison
N_BINS = 15


# ---------------------------------------------------------------------------
# Data / model setup — mirrors evaluate.py's pattern so the val/test split
# and metadata encoding are byte-for-byte identical to what training used.
# ---------------------------------------------------------------------------

def build_datasets(config, metadata_csv: str, images_dir_part1: str, images_dir_part2: str):
    df = load_metadata(metadata_csv)
    train_df, val_df, test_df = stratified_split(
        df,
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
        config["data"]["random_seed"],
    )
    image_dirs = [images_dir_part1, images_dir_part2]
    image_size = config["train"]["image_size"]
    use_metadata = config["train"].get("use_metadata", False)

    # Train set is only built to recover the age_scaler / age_median /
    # metadata_columns fitted during training — its images are never used here.
    train_dataset = HAM10000Dataset(
        train_df, image_dirs, config["data"]["image_extension"],
        transform=get_eval_transforms(image_size),
        use_metadata=use_metadata,
    )
    age_scaler = train_dataset.age_scaler if use_metadata else None
    age_median = train_dataset.age_median if use_metadata else None
    metadata_columns = train_dataset.metadata_columns if use_metadata else None
    if use_metadata:
        config["model"]["metadata_dim"] = len(metadata_columns)

    def make_eval_dataset(split_df):
        return HAM10000Dataset(
            split_df, image_dirs, config["data"]["image_extension"],
            transform=get_eval_transforms(image_size),
            use_metadata=use_metadata,
            age_scaler=age_scaler,
            age_median=age_median,
            metadata_columns=metadata_columns,
        )

    val_dataset = make_eval_dataset(val_df)
    test_dataset = make_eval_dataset(test_df)
    return val_dataset, test_dataset, use_metadata


@torch.no_grad()
def collect_logits(model, dataset, device, use_metadata, batch_size, num_workers):
    """Runs a forward pass over `dataset` and returns (logits, labels) as CPU tensors."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_logits, all_labels = [], []
    for batch in loader:
        if use_metadata:
            images, metas, labels = batch
            images, metas = images.to(device), metas.to(device)
            outputs = model(images, metas)
        else:
            images, labels = batch
            images = images.to(device)
            outputs = model(images)
        all_logits.append(outputs.detach().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def load_or_compute(cache_dir: Path, name: str, compute_fn, force_recompute: bool):
    logits_path = cache_dir / f"{name}_logits.pt"
    labels_path = cache_dir / f"{name}_labels.pt"
    if not force_recompute and logits_path.exists() and labels_path.exists():
        print(f"[calibrate] Using cached {name} logits/labels from {cache_dir}")
        return torch.load(logits_path), torch.load(labels_path)

    logits, labels = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(logits, logits_path)
    torch.save(labels, labels_path)
    print(f"[calibrate] Cached {name} logits/labels ({logits.shape[0]} samples) to {cache_dir}")
    return logits, labels


# ---------------------------------------------------------------------------
# Temperature fitting (Guo et al. 2017, section 4)
# ---------------------------------------------------------------------------

def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
    """
    Fits a single scalar T by minimizing NLL of (val_logits / T) against
    val_labels, using LBFGS. T is parameterized as exp(log_T) so it can
    never go negative or hit exactly zero, and is initialized at T=1.0
    (log_T=0), i.e. the uncalibrated starting point.
    """
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(val_logits / T, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_T = log_T.exp().item()
    return fitted_T


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece_mce(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """
    Standard equal-width-bin ECE/MCE over the *predicted-class* confidence
    (i.e. max softmax probability), against whether that prediction was
    correct — the formulation from Guo et al. 2017, eq. 3.
    Returns (ece, mce, per_bin) where per_bin holds, for each bin: the bin's
    average confidence, average accuracy, and sample count (for plotting).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    mce = 0.0
    per_bin = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count == 0:
            per_bin.append((None, None, 0))
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        per_bin.append((bin_conf, bin_acc, int(count)))
    return ece, mce, per_bin


def evaluate_at_temperature(logits: torch.Tensor, labels: torch.Tensor, T: float):
    probs = F.softmax(logits / T, dim=1)
    confidences, preds = probs.max(dim=1)
    correct = (preds == labels).numpy().astype(float)
    confidences = confidences.numpy()

    nll = F.cross_entropy(logits / T, labels).item()
    acc = accuracy_score(labels.numpy(), preds.numpy())
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    ece, mce, per_bin = compute_ece_mce(confidences, correct, N_BINS)

    return {
        "T": T,
        "nll": nll,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": ece,
        "mce": mce,
        "per_bin": per_bin,
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(uncal_result, cal_result, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / N_BINS

    for ax, result, title in [
        (axes[0], uncal_result, f"Uncalibrated (T=1.0)\nECE={uncal_result['ece']:.4f}"),
        (axes[1], cal_result, f"Fitted T={cal_result['T']:.4f}\nECE={cal_result['ece']:.4f}"),
    ]:
        accs = [b[1] if b[1] is not None else 0.0 for b in result["per_bin"]]
        ax.bar(bin_centers, accs, width=bin_width * 0.9, color="#3b6ea5", edgecolor="black",
               label="Accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Reliability diagram: confidence vs. accuracy (15 equal-width bins)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[calibrate] Reliability diagram saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling on the trained model (Guo et al. 2017)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--metadata-csv", type=str, default="data/HAM10000_metadata.csv",
                         help="Overrides config's data.metadata_csv (that path is Kaggle-only).")
    parser.add_argument("--images-dir-part1", type=str, required=True)
    parser.add_argument("--images-dir-part2", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="reports/calibration/cache")
    parser.add_argument("--output-dir", type=str, default="reports/calibration")
    parser.add_argument("--force-recompute", action="store_true",
                         help="Ignore cached logits/labels and re-run inference.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["train"]["device"])
    # get_device() (src/utils.py, shared with train.py/evaluate.py) only checks
    # for CUDA and falls back to CPU otherwise. Left untouched deliberately —
    # this script instead opportunistically upgrades a CPU fallback to MPS
    # locally, since that's a large speedup on Apple Silicon and doesn't change
    # any shared code path.
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[calibrate] CUDA unavailable; using MPS (Apple Silicon) instead of CPU.")
    print(f"[calibrate] Using device: {device}")

    val_dataset, test_dataset, use_metadata = build_datasets(
        config, args.metadata_csv, args.images_dir_part1, args.images_dir_part2
    )

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    cache_dir = Path(args.cache_dir)

    # ---- Step 2: validation logits (for fitting T) ----
    val_logits, val_labels = load_or_compute(
        cache_dir, "val",
        lambda: collect_logits(model, val_dataset, device, use_metadata, batch_size, num_workers),
        args.force_recompute,
    )

    # ---- Step 3: fit T on validation logits ----
    fitted_T = fit_temperature(val_logits, val_labels)
    print(f"\n[calibrate] Fitted temperature T = {fitted_T:.4f}\n")

    # ---- Step 4: test logits, evaluate all three T values on TEST ----
    test_logits, test_labels = load_or_compute(
        cache_dir, "test",
        lambda: collect_logits(model, test_dataset, device, use_metadata, batch_size, num_workers),
        args.force_recompute,
    )

    results = {
        "T=1.0 (uncalibrated)": evaluate_at_temperature(test_logits, test_labels, 1.0),
        f"T={HAND_PICKED_T} (current app.py)": evaluate_at_temperature(test_logits, test_labels, HAND_PICKED_T),
        f"T={fitted_T:.4f} (fitted)": evaluate_at_temperature(test_logits, test_labels, fitted_T),
    }

    # ---- Sanity check: accuracy/macro-F1 must be identical across T ----
    accs = {k: r["accuracy"] for k, r in results.items()}
    f1s = {k: r["macro_f1"] for k, r in results.items()}
    if len(set(round(a, 10) for a in accs.values())) > 1 or len(set(round(f, 10) for f in f1s.values())) > 1:
        print("\n[calibrate] STOP: accuracy or macro-F1 differs across temperatures.")
        print(f"  accuracies: {accs}")
        print(f"  macro_f1s:  {f1s}")
        print(
            "  Temperature scaling divides all logits by a positive scalar, which cannot "
            "change argmax — this means something upstream (e.g. logits/labels misaligned "
            "between runs, or a non-deterministic model in eval mode) is wrong. Not "
            "explaining this away — investigate before trusting any ECE/NLL numbers above."
        )
        raise SystemExit(1)

    # ---- Report table ----
    print("=" * 78)
    print(f"{'':28s}{'ECE':>10s}{'MCE':>10s}{'NLL':>10s}{'Top-1 Acc':>12s}{'Macro-F1':>10s}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:28s}{r['ece']:>10.4f}{r['mce']:>10.4f}{r['nll']:>10.4f}{r['accuracy']:>12.4f}{r['macro_f1']:>10.4f}")
    print("=" * 78)

    # ---- Reliability diagram: uncalibrated vs fitted-T ----
    out_dir = Path(args.output_dir)
    plot_reliability_diagrams(
        results["T=1.0 (uncalibrated)"],
        results[f"T={fitted_T:.4f} (fitted)"],
        out_dir / "reliability_diagram.png",
    )

    # ---- Plain summary ----
    uncal_ece = results["T=1.0 (uncalibrated)"]["ece"]
    handpicked_ece = results[f"T={HAND_PICKED_T} (current app.py)"]["ece"]
    fitted_ece = results[f"T={fitted_T:.4f} (fitted)"]["ece"]
    beats_handpicked = fitted_ece < handpicked_ece
    delta = handpicked_ece - fitted_ece

    print("\nSUMMARY")
    print("-" * 78)
    print(f"Fitted T (on validation set, NLL-minimizing): {fitted_T:.4f}")
    print(f"Test ECE — uncalibrated (T=1.0):       {uncal_ece:.4f}")
    print(f"Test ECE — current hand-picked (T={HAND_PICKED_T}): {handpicked_ece:.4f}")
    print(f"Test ECE — fitted (T={fitted_T:.4f}):          {fitted_ece:.4f}")
    if beats_handpicked:
        print(f"-> Fitted T beats the current hand-picked T={HAND_PICKED_T} on test ECE by {delta:.4f} "
              f"({delta / handpicked_ece * 100:.1f}% relative reduction).")
    else:
        print(f"-> Fitted T does NOT beat the current hand-picked T={HAND_PICKED_T} on test ECE "
              f"(worse by {-delta:.4f}).")
    print(
        "\nNo files outside reports/ were touched. app.py's TEMPERATURE constant was NOT "
        "changed — this script only reports numbers for you to decide on."
    )


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"[FILE ERROR] {e}")
        raise SystemExit(1)
