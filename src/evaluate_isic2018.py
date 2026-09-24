"""
External validation on the official ISIC 2018 Task 3 test set.

Evaluation only: never trains, fine-tunes, or refits the temperature or any
threshold. TEMPERATURE below is a fixed value copied from app.py — this
script does NOT call src.calibrate.fit_temperature.

Data provenance (see the accompanying report for the full audit):
- Images:     ISIC2018_Task3_Test_Images.zip, Harvard Dataverse doi:10.7910/DVN/DBW86T,
              file id 3855824 ("Test-Set images of the ISIC 2018 challenge (Task 3)").
- Ground truth + metadata: ISIC2018_Task3_Test_GroundTruth.tab, same Dataverse record,
              file id 6924466 ("Ground truth for the ISIC2018 Task 3 challenge, in the
              same format as the HAM10000 metadata"). Columns: lesion_id, image_id, dx,
              dx_type, age, sex, localization, dataset — i.e. per-image age/sex/
              localization IS included, released after the original 2018 challenge
              (this dataset record's version 4 was released 2023-02-07).
- The ground truth file has 1,512 rows, but the images zip contains only 1,511 .jpg
  files — image_id "ISIC_0035068" (dx=nv, all metadata fields empty in the ground
  truth) has no corresponding image in the released archive. HAM10000Dataset's
  existing missing-image skip-and-warn logic drops it automatically, leaving 1,511
  evaluable images — this is the source of the "1,511 vs 1,512" discrepancy seen
  across the literature for this dataset.
- These images/labels are NOT committed to this repo. Point --isic-images-dir and
  --isic-groundtruth at wherever you downloaded them locally.

Usage:
    python -m src.evaluate_isic2018 \
        --checkpoint models/best_model.pt \
        --ham-images-dir-part1 /path/to/HAM10000_images_part_1 \
        --ham-images-dir-part2 /path/to/HAM10000_images_part_2 \
        --isic-images-dir /path/to/ISIC2018_Task3_Test_Images \
        --isic-groundtruth /path/to/ISIC2018_Task3_Test_GroundTruth.tab
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import torch
import torch.nn.functional as F

from src.calibrate import build_datasets, collect_logits, evaluate_at_temperature, load_or_compute
from src.config import ConfigError, load_config
from src.dataset import DX_LABELS, HAM10000Dataset, load_metadata
from src.error_analysis import (
    MALIGNANT_CLASSES,
    analyze_class,
    per_class_table,
    print_confusion_matrix,
    print_per_class_table,
    save_confusion_matrix_png,
)
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint

# Must match app.py's TEMPERATURE constant (fitted on the validation set in
# src/calibrate.py). NOT refit here — this script only reports whether that
# fitted value still calibrates well on an external test set.
CURRENT_TEMPERATURE = 2.1235

N_BOOTSTRAP_DEFAULT = 1000
CI_PERCENTILES = (2.5, 97.5)

MALIGNANT_IDX = [DX_LABELS.index(c) for c in MALIGNANT_CLASSES]


# ---------------------------------------------------------------------------
# Data acquisition helpers
# ---------------------------------------------------------------------------

def load_isic_groundtruth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", na_values=["", "NA"])
    required = {"lesion_id", "image_id", "dx", "age", "sex", "localization"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ISIC ground truth file is missing expected column(s): {missing}")
    return df


def _find_image_path(image_id: str, image_dirs: list[Path], ext: str = ".jpg") -> Path | None:
    for d in image_dirs:
        candidate = d / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leakage_check(
    ham_df: pd.DataFrame,
    isic_df: pd.DataFrame,
    ham_image_dirs: list[Path],
    isic_image_dir: Path,
) -> None:
    """
    Confirms the ISIC 2018 Task 3 test set does not duplicate anything in the
    local HAM10000 training data: (1) image_id overlap, (2) lesion_id overlap,
    (3) exact byte-identical image content (MD5) — since (1)/(2) only catch
    duplicates that kept the same ID, not a re-uploaded copy under a new one.
    Prints a report; raises if any overlap is found, since evaluating on
    leaked data would silently overstate performance.
    """
    print("=" * 70)
    print("LEAKAGE CHECK (HAM10000 train/val/test pool vs. ISIC 2018 test set)")
    print("=" * 70)

    ham_image_ids = set(ham_df["image_id"])
    ham_lesion_ids = set(ham_df["lesion_id"])
    isic_image_ids = set(isic_df["image_id"])
    isic_lesion_ids = set(isic_df["lesion_id"])

    image_id_overlap = ham_image_ids & isic_image_ids
    lesion_id_overlap = ham_lesion_ids & isic_lesion_ids
    print(f"image_id overlap:  {len(image_id_overlap)} (of {len(isic_image_ids)} test image_ids)")
    print(f"lesion_id overlap: {len(lesion_id_overlap)} (of {len(isic_lesion_ids)} test lesion_ids)")

    print("Hashing images for exact byte-identical content check (MD5) ...")
    ham_hashes: dict[str, str] = {}
    for image_id in ham_df["image_id"]:
        p = _find_image_path(image_id, ham_image_dirs)
        if p is not None:
            ham_hashes[image_id] = _md5(p)

    isic_hashes: dict[str, str] = {}
    for image_id in isic_df["image_id"]:
        p = _find_image_path(image_id, [isic_image_dir])
        if p is not None:
            isic_hashes[image_id] = _md5(p)

    ham_hash_to_ids: dict[str, list[str]] = {}
    for image_id, h in ham_hashes.items():
        ham_hash_to_ids.setdefault(h, []).append(image_id)

    content_overlap = [
        (image_id, h, ham_hash_to_ids[h])
        for image_id, h in isic_hashes.items()
        if h in ham_hash_to_ids
    ]
    print(f"exact byte-identical image overlap (MD5): {len(content_overlap)} (of {len(isic_hashes)} hashed test images)")
    for image_id, h, matches in content_overlap[:20]:
        print(f"  test {image_id} (md5={h}) == train {matches}")

    total_overlap = len(image_id_overlap) + len(lesion_id_overlap) + len(content_overlap)
    if total_overlap > 0:
        raise RuntimeError(
            f"Leakage check FAILED: {len(image_id_overlap)} image_id, "
            f"{len(lesion_id_overlap)} lesion_id, {len(content_overlap)} exact-content "
            f"overlaps found between the local HAM10000 pool and the ISIC 2018 test set. "
            f"Evaluating on this data would overstate performance — not proceeding."
        )
    print("No overlap found by image_id, lesion_id, or exact image content (MD5).")
    print(
        "Note: MD5 only catches byte-identical files; it would not catch a "
        "re-compressed/re-scaled copy of the same photo. lesion_id namespaces "
        "are disjoint ('HAM_*' train vs 'HAMTEST_*' test), which is the stronger signal."
    )
    print()


# ---------------------------------------------------------------------------
# ISIC 2024 official primary metric: partial AUC above a minimum TPR.
# Ported verbatim (only renamed) from ISIC-Research/Challenge-2024-Metrics,
# PrimaryMetric-pAUC.py, (c) 2024 Nicholas R Kurtansky, MSKCC. Returns a raw
# (non-rescaled) area, so for min_tpr=0.80 the value range is [0, 0.2].
# ---------------------------------------------------------------------------

def partial_auc_above_tpr(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80) -> float:
    if len(np.unique(y_true)) != 2:
        return float("nan")
    v_gt = np.abs(np.asarray(y_true) - 1)
    v_pred = np.abs(np.asarray(y_score) - 1)
    max_fpr = abs(1 - min_tpr)

    fpr, tpr, _ = skmetrics.roc_curve(v_gt, v_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop == 0 or stop >= len(fpr):
        return float("nan")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return float(skmetrics.auc(fpr, tpr))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(n: int, metric_fn, n_boot: int = N_BOOTSTRAP_DEFAULT, seed: int = 42):
    """
    Percentile bootstrap over `n` samples. `metric_fn(idx)` computes the metric
    given an index array (with replacement for resamples). Returns
    (point_estimate, ci_lo, ci_hi); point estimate is metric_fn on the
    original (unresampled) indices. Resamples where metric_fn returns NaN
    (e.g. a class absent from that resample) are dropped from the CI.
    """
    point = metric_fn(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(idx)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            boots.append(val)
    if len(boots) < n_boot // 2:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, CI_PERCENTILES)
    return point, float(lo), float(hi)


def fmt_ci(point: float, lo: float, hi: float, digits: int = 4) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------------------
# Metrics bundle, computed identically for the external and internal sets
# ---------------------------------------------------------------------------

def full_report(name: str, logits: torch.Tensor, labels: torch.Tensor, output_dir: Path, n_boot: int, seed: int):
    labels_np = labels.numpy()
    preds_np = logits.argmax(dim=1).numpy()
    probs_t1 = F.softmax(logits, dim=1).numpy()
    n = len(labels_np)

    print("\n" + "#" * 70)
    print(f"# {name}  (n={n})")
    print("#" * 70)

    accuracy_pt, accuracy_lo, accuracy_hi = bootstrap_ci(
        n, lambda idx: skmetrics.accuracy_score(labels_np[idx], preds_np[idx]), n_boot, seed
    )
    bal_acc_pt, bal_acc_lo, bal_acc_hi = bootstrap_ci(
        n, lambda idx: skmetrics.balanced_accuracy_score(labels_np[idx], preds_np[idx]), n_boot, seed
    )
    macro_f1_pt, macro_f1_lo, macro_f1_hi = bootstrap_ci(
        n, lambda idx: skmetrics.f1_score(labels_np[idx], preds_np[idx], average="macro", zero_division=0),
        n_boot, seed,
    )
    print(f"Accuracy:                    {fmt_ci(accuracy_pt, accuracy_lo, accuracy_hi)}")
    print(f"Balanced multi-class acc.:   {fmt_ci(bal_acc_pt, bal_acc_lo, bal_acc_hi)}  (mean per-class recall — ISIC 2018 leaderboard metric)")
    print(f"Macro-F1:                    {fmt_ci(macro_f1_pt, macro_f1_lo, macro_f1_hi)}")

    rows = per_class_table(labels_np, preds_np)
    print_per_class_table(rows)

    cm = skmetrics.confusion_matrix(labels_np, preds_np, labels=range(len(DX_LABELS)))
    print_confusion_matrix(cm)
    save_confusion_matrix_png(cm, output_dir / f"{name}_confusion_matrix.png")

    confidences_calibrated = F.softmax(logits / CURRENT_TEMPERATURE, dim=1).max(dim=1).values.numpy()
    mel_idx = DX_LABELS.index("mel")
    mel_result = analyze_class("mel", labels_np, preds_np, confidences_calibrated, verbose=False)

    def _mel_recall(idx):
        support = int((labels_np[idx] == mel_idx).sum())
        if support == 0:
            return float("nan")
        tp = int(((labels_np[idx] == mel_idx) & (preds_np[idx] == mel_idx)).sum())
        return tp / support

    def _mel_precision(idx):
        predicted_pos = int((preds_np[idx] == mel_idx).sum())
        if predicted_pos == 0:
            return float("nan")
        tp = int(((labels_np[idx] == mel_idx) & (preds_np[idx] == mel_idx)).sum())
        return tp / predicted_pos

    mel_recall_pt, mel_recall_lo, mel_recall_hi = bootstrap_ci(n, _mel_recall, n_boot, seed)
    mel_prec_pt, mel_prec_lo, mel_prec_hi = bootstrap_ci(n, _mel_precision, n_boot, seed)
    print(f"\nMelanoma recall (with CI):    {fmt_ci(mel_recall_pt, mel_recall_lo, mel_recall_hi)}")
    print(f"Melanoma precision (with CI): {fmt_ci(mel_prec_pt, mel_prec_lo, mel_prec_hi)}")

    mal_labels = np.isin(labels_np, MALIGNANT_IDX).astype(int)
    mal_scores = probs_t1[:, MALIGNANT_IDX].sum(axis=1)

    roc_auc_pt, roc_auc_lo, roc_auc_hi = bootstrap_ci(
        n, lambda idx: skmetrics.roc_auc_score(mal_labels[idx], mal_scores[idx])
        if len(np.unique(mal_labels[idx])) == 2 else float("nan"),
        n_boot, seed,
    )
    pauc_pt, pauc_lo, pauc_hi = bootstrap_ci(
        n, lambda idx: partial_auc_above_tpr(mal_labels[idx], mal_scores[idx], min_tpr=0.80),
        n_boot, seed,
    )
    print(f"\nMalignant(mel/bcc/akiec) vs benign ROC-AUC:          {fmt_ci(roc_auc_pt, roc_auc_lo, roc_auc_hi)}")
    print(f"pAUC above 80% TPR (ISIC 2024 definition, range 0-0.2): {fmt_ci(pauc_pt, pauc_lo, pauc_hi)}")

    calib_t1 = evaluate_at_temperature(logits, labels, 1.0)
    calib_tcur = evaluate_at_temperature(logits, labels, CURRENT_TEMPERATURE)
    print(f"\nCalibration @ T=1.0:              ECE={calib_t1['ece']:.4f}  MCE={calib_t1['mce']:.4f}  NLL={calib_t1['nll']:.4f}")
    print(f"Calibration @ T={CURRENT_TEMPERATURE} (current app.py): ECE={calib_tcur['ece']:.4f}  MCE={calib_tcur['mce']:.4f}  NLL={calib_tcur['nll']:.4f}")

    return {
        "n": n,
        "accuracy": (accuracy_pt, accuracy_lo, accuracy_hi),
        "balanced_accuracy": (bal_acc_pt, bal_acc_lo, bal_acc_hi),
        "macro_f1": (macro_f1_pt, macro_f1_lo, macro_f1_hi),
        "mel_recall": (mel_recall_pt, mel_recall_lo, mel_recall_hi),
        "mel_precision": (mel_prec_pt, mel_prec_lo, mel_prec_hi),
        "roc_auc": (roc_auc_pt, roc_auc_lo, roc_auc_hi),
        "pauc_80tpr": (pauc_pt, pauc_lo, pauc_hi),
        "ece_t1": calib_t1["ece"], "mce_t1": calib_t1["mce"], "nll_t1": calib_t1["nll"],
        "ece_tcur": calib_tcur["ece"], "mce_tcur": calib_tcur["mce"], "nll_tcur": calib_tcur["nll"],
        "mel_missed": mel_result["n_missed"],
        "mel_support": mel_result["support"],
        "mel_misroutes": mel_result["misroute_counts"],
    }


def comparison_table(internal: dict, external: dict):
    print("\n" + "=" * 78)
    print("COMPARISON: internal HAM10000 test set vs. official ISIC 2018 Task 3 test set")
    print("=" * 78)
    print(f"{'metric':30s}{'internal (n=' + str(internal['n']) + ')':>22s}{'external (n=' + str(external['n']) + ')':>26s}")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced multi-class acc."),
        ("macro_f1", "Macro-F1"),
        ("mel_recall", "Melanoma recall"),
        ("mel_precision", "Melanoma precision"),
        ("roc_auc", "Malignant ROC-AUC"),
        ("pauc_80tpr", "pAUC @80% TPR"),
    ]:
        i_pt, i_lo, i_hi = internal[key]
        e_pt, e_lo, e_hi = external[key]
        outside = not (e_lo <= i_pt <= e_hi) if not np.isnan(e_lo) else True
        verdict = "OUTSIDE external 95% CI" if outside else "within external 95% CI (consistent with noise)"
        print(f"{label:30s}{i_pt:>10.4f}{'':<12s}{fmt_ci(e_pt, e_lo, e_hi):>26s}")
        print(f"{'':30s}delta={e_pt - i_pt:+.4f}  -> internal point is {verdict}")
    print()
    print(
        f"Melanoma misses: internal {internal['mel_missed']}/{internal['mel_support']} vs. "
        f"external {external['mel_missed']}/{external['mel_support']}."
    )
    print(f"Internal missed-melanoma routed to: {internal['mel_misroutes']}")
    print(f"External missed-melanoma routed to: {external['mel_misroutes']}")

    print("\nCalibration transfer (does T=2.1235, fitted on our validation set, still hold externally?):")
    print(f"  internal @ T=1.0:    ECE={internal['ece_t1']:.4f}  MCE={internal['mce_t1']:.4f}  NLL={internal['nll_t1']:.4f}")
    print(f"  internal @ T={CURRENT_TEMPERATURE}: ECE={internal['ece_tcur']:.4f}  MCE={internal['mce_tcur']:.4f}  NLL={internal['nll_tcur']:.4f}")
    print(f"  external @ T=1.0:    ECE={external['ece_t1']:.4f}  MCE={external['mce_t1']:.4f}  NLL={external['nll_t1']:.4f}")
    print(f"  external @ T={CURRENT_TEMPERATURE}: ECE={external['ece_tcur']:.4f}  MCE={external['mce_tcur']:.4f}  NLL={external['nll_tcur']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="External evaluation on the official ISIC 2018 Task 3 test set (read-only, no fitting).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--ham-metadata-csv", type=str, default="data/HAM10000_metadata.csv")
    parser.add_argument("--ham-images-dir-part1", type=str, required=True)
    parser.add_argument("--ham-images-dir-part2", type=str, required=True)
    parser.add_argument("--isic-images-dir", type=str, required=True)
    parser.add_argument("--isic-groundtruth", type=str, required=True)
    parser.add_argument("--internal-cache-dir", type=str, default="reports/calibration/cache")
    parser.add_argument("--cache-dir", type=str, default="reports/isic2018_test/cache")
    parser.add_argument("--output-dir", type=str, default="reports/isic2018_test")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ham_df = load_metadata(args.ham_metadata_csv)
    isic_df = load_isic_groundtruth(args.isic_groundtruth)

    leakage_check(
        ham_df, isic_df,
        [Path(args.ham_images_dir_part1), Path(args.ham_images_dir_part2)],
        Path(args.isic_images_dir),
    )

    device = get_device(config["train"]["device"])
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[evaluate_isic2018] CUDA unavailable; using MPS instead of CPU.")

    # Reuse calibrate.py's exact train/val/test split + age_scaler/metadata_columns
    # fitting path, so the ISIC test set is encoded byte-for-byte the same way
    # app.py encodes a live request.
    _, ham_test_dataset, use_metadata = build_datasets(
        config, args.ham_metadata_csv, args.ham_images_dir_part1, args.ham_images_dir_part2
    )
    age_scaler = ham_test_dataset.age_scaler
    age_median = ham_test_dataset.age_median
    metadata_columns = ham_test_dataset.metadata_columns
    if use_metadata:
        config["model"]["metadata_dim"] = len(metadata_columns)

    isic_dataset = HAM10000Dataset(
        isic_df, [args.isic_images_dir], ".jpg",
        transform=get_eval_transforms(config["train"]["image_size"]),
        use_metadata=use_metadata,
        age_scaler=age_scaler,
        age_median=age_median,
        metadata_columns=metadata_columns,
    )
    print(f"[evaluate_isic2018] ISIC 2018 test set: {len(isic_dataset)} evaluable images "
          f"(of {len(isic_df)} ground-truth rows).")

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    cache_dir = Path(args.cache_dir)
    external_logits, external_labels = load_or_compute(
        cache_dir, "isic2018_test",
        lambda: collect_logits(model, isic_dataset, device, use_metadata,
                                config["train"]["batch_size"], config["train"]["num_workers"]),
        args.force_recompute,
    )

    internal_cache = Path(args.internal_cache_dir)
    internal_logits = torch.load(internal_cache / "test_logits.pt")
    internal_labels = torch.load(internal_cache / "test_labels.pt")

    internal_report = full_report("internal_ham10000_test", internal_logits, internal_labels,
                                   output_dir, args.n_bootstrap, args.seed)
    external_report = full_report("external_isic2018_test", external_logits, external_labels,
                                   output_dir, args.n_bootstrap, args.seed)

    comparison_table(internal_report, external_report)


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"[FILE ERROR] {e}")
        raise SystemExit(1)
