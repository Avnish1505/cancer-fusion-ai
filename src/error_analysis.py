"""
Phase 4 evaluation deep-dive: per-class error analysis on the TEST set,
with melanoma (and the other malignant/pre-malignant classes) called out
explicitly rather than averaged into macro-F1.

Read-only: does not touch the model, checkpoint, thresholds, or training.
Reuses the test logits/labels cached by src/calibrate.py
(reports/calibration/cache/test_{logits,labels}.pt) instead of re-running
inference.

Usage:
    python -m src.error_analysis
    python -m src.error_analysis --cache-dir reports/calibration/cache --temperature 2.1235
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.dataset import DX_LABELS

# Matches the TEMPERATURE now adopted in app.py (fitted via src/calibrate.py).
# Only affects the *reported confidence* numbers below (section 4) — it has
# no effect on predictions, precision/recall/F1, or the confusion matrix,
# since argmax is invariant to dividing logits by a positive scalar.
DEFAULT_TEMPERATURE = 2.1235

MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]  # melanoma, basal cell carcinoma, actinic keratoses/Bowen's


def load_cached_test_set(cache_dir: Path):
    logits_path = cache_dir / "test_logits.pt"
    labels_path = cache_dir / "test_labels.pt"
    if not logits_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing cached test logits/labels in '{cache_dir}'. Run "
            f"`python -m src.calibrate ...` first (or check --cache-dir)."
        )
    logits = torch.load(logits_path)
    labels = torch.load(labels_path)
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"Cached test logits ({logits.shape[0]} rows) and labels "
            f"({labels.shape[0]} rows) don't line up — cache looks corrupt "
            f"or was written by two different runs. Re-run src.calibrate."
        )
    return logits, labels


def per_class_table(labels: np.ndarray, preds: np.ndarray) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(DX_LABELS)), zero_division=0
    )
    rows = [
        {
            "class": DX_LABELS[i],
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }
        for i in range(len(DX_LABELS))
    ]
    rows.sort(key=lambda r: r["recall"])
    return rows


def print_per_class_table(rows: list[dict]):
    print("=" * 62)
    print("PER-CLASS METRICS (test set, sorted by recall ascending)")
    print("=" * 62)
    print(f"{'class':8s}{'precision':>12s}{'recall':>10s}{'f1':>10s}{'support':>10s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['class']:8s}{r['precision']:>12.4f}{r['recall']:>10.4f}{r['f1']:>10.4f}{r['support']:>10d}")
    print("=" * 62)


def print_confusion_matrix(cm: np.ndarray):
    print("\nCONFUSION MATRIX (rows=true, cols=predicted)")
    header = "true\\pred".ljust(10) + "".join(f"{c:>7s}" for c in DX_LABELS)
    print(header)
    for i, row_label in enumerate(DX_LABELS):
        row_str = row_label.ljust(10) + "".join(f"{cm[i, j]:>7d}" for j in range(len(DX_LABELS)))
        print(row_str)


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path):
    # Plain matplotlib (no seaborn) — keeps this read-only script's
    # dependencies limited to what's already in requirements.txt.
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(DX_LABELS)))
    ax.set_yticks(range(len(DX_LABELS)))
    ax.set_xticklabels(DX_LABELS)
    ax.set_yticklabels(DX_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"HAM10000 test set — confusion matrix (n={cm.sum()})")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[error_analysis] Confusion matrix saved to {out_path}")


def analyze_class(class_name: str, labels: np.ndarray, preds: np.ndarray, confidences: np.ndarray, verbose: bool):
    idx = DX_LABELS.index(class_name)
    is_true = labels == idx
    support = int(is_true.sum())
    true_positive = int(((labels == idx) & (preds == idx)).sum())
    predicted_positive = int((preds == idx).sum())

    recall = true_positive / support if support > 0 else float("nan")
    precision = true_positive / predicted_positive if predicted_positive > 0 else float("nan")

    missed_mask = is_true & (preds != idx)
    n_missed = int(missed_mask.sum())
    missed_preds = preds[missed_mask]
    missed_confidences = confidences[missed_mask]

    # Where the missed cases actually landed, most-common first.
    misroute_counts = {}
    for p in missed_preds:
        misroute_counts[DX_LABELS[p]] = misroute_counts.get(DX_LABELS[p], 0) + 1
    misroute_counts = dict(sorted(misroute_counts.items(), key=lambda kv: -kv[1]))

    print(f"\n{'-' * 62}")
    print(class_name.upper())
    print(f"{'-' * 62}")
    print(f"  True {class_name.upper()} cases in test set : {support}")
    print(f"  Recall                          : {recall:.4f}")
    print(f"  Precision                       : {precision:.4f}")
    print(f"  Missed (false negatives)        : {n_missed} of {support}")
    if n_missed > 0:
        print(f"  Missed cases were predicted as  : {misroute_counts}")
        print(
            f"  Confidence on the (wrong) top prediction, missed cases only "
            f"(T={DEFAULT_TEMPERATURE} calibrated):"
        )
        print(
            f"    mean={missed_confidences.mean():.4f}  median={np.median(missed_confidences):.4f}  "
            f"min={missed_confidences.min():.4f}  max={missed_confidences.max():.4f}"
        )
        if verbose:
            sorted_conf = np.sort(missed_confidences)[::-1]
            print(f"    all missed-case confidences (desc): {np.round(sorted_conf, 4).tolist()}")
    else:
        print("  Missed cases were predicted as  : (none — recall is 1.0)")

    return {
        "support": support,
        "recall": recall,
        "precision": precision,
        "n_missed": n_missed,
        "misroute_counts": misroute_counts,
        "missed_confidences": missed_confidences,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 error analysis on cached test-set logits")
    parser.add_argument("--cache-dir", type=str, default="reports/calibration/cache")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                         help="Only affects reported confidence values, never predictions.")
    args = parser.parse_args()

    logits, labels_t = load_cached_test_set(Path(args.cache_dir))
    labels = labels_t.numpy()

    # argmax is invariant to dividing by a positive scalar, so predictions
    # (and therefore precision/recall/F1/confusion matrix) do not depend on
    # temperature. Only the reported confidence values (section 4/5) do.
    preds = logits.argmax(dim=1).numpy()
    probs = F.softmax(logits / args.temperature, dim=1)
    confidences = probs.max(dim=1).values.numpy()

    n = len(labels)
    print(f"[error_analysis] Loaded {n} cached test-set predictions from {args.cache_dir}")

    # ---- 1. Per-class precision/recall/F1/support ----
    rows = per_class_table(labels, preds)
    print_per_class_table(rows)

    # ---- 2. Confusion matrix ----
    cm = confusion_matrix(labels, preds, labels=range(len(DX_LABELS)))
    print_confusion_matrix(cm)
    save_confusion_matrix_png(cm, Path(args.output_dir) / "confusion_matrix.png")

    # ---- 3 & 4. Melanoma, in full detail ----
    print("\n" + "=" * 62)
    print("MALIGNANT / PRE-MALIGNANT CLASSES — CALLED OUT EXPLICITLY")
    print("(never averaged into macro-F1 — a missed melanoma is not the")
    print(" same kind of error as a missed vascular lesion)")
    print("=" * 62)
    mel_result = analyze_class("mel", labels, preds, confidences, verbose=True)

    # ---- 5. BCC and AKIEC, less detail ----
    bcc_result = analyze_class("bcc", labels, preds, confidences, verbose=False)
    akiec_result = analyze_class("akiec", labels, preds, confidences, verbose=False)

    # ---- Plain-language summary ----
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    mel_recall_pct = mel_result["recall"] * 100
    print(
        f"Melanoma recall on the test set: {mel_recall_pct:.1f}% "
        f"({mel_result['support'] - mel_result['n_missed']} of {mel_result['support']} true "
        f"melanoma cases caught; {mel_result['n_missed']} missed)."
    )
    if mel_result["n_missed"] > 0:
        mean_conf = mel_result["missed_confidences"].mean()
        conf_word = "confidently" if mean_conf >= 0.5 else "hesitantly"
        print(
            f"On the melanoma cases it missed, the model's average confidence in its "
            f"(wrong) top prediction was {mean_conf:.1%} — it tends to miss melanoma "
            f"{conf_word}, not with a clear 'unsure' signal in most cases."
        )
    print()
    print(
        "Assessment: a melanoma recall of "
        f"{mel_recall_pct:.1f}% means roughly "
        f"{'more than 1 in 5' if mel_recall_pct < 80 else ('about 1 in 10' if mel_recall_pct < 90 else 'fewer than 1 in 10')} "
        "true melanomas in this test set would be missed if this model's top-1 prediction "
        "were relied on alone. For a research prototype that is explicitly not a diagnostic "
        "device, that is a plausible starting point to iterate from — but it is NOT "
        "acceptable as a stand-alone melanoma screen, and the false negatives above show it "
        "sometimes misses with real confidence rather than visible hesitation, which is the "
        "more dangerous failure mode of the two. Any clinical framing of this tool would need "
        "this number to be much higher (and ideally reported with a confidence interval over a "
        "larger test set — n=1543 total, with far fewer true melanoma cases, means this "
        "estimate has real sampling noise)."
    )


if __name__ == "__main__":
    main()
