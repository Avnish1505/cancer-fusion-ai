"""
Exports frozen inference-time artifacts to models/inference_artifacts.json
(+ models/mahalanobis_params.npz), so app.py never hardcodes an analysis
number and can verify at startup that those numbers still match the
checkpoint they were computed from.

Every value here is either read directly from an existing analysis cache
(reports/calibration/cache/, reports/ood/cache/) or recomputed by the exact
function the analysis scripts use — src.calibrate.fit_temperature,
src.conformal.lac_scores/calibrate_mondrian/malignant_prob/choose_threshold,
src.ood.fit_mahalanobis/accept_threshold_low_is_ood — nothing is typed in
by hand. Re-run this whenever models/best_model.pt changes.

Usage:
    python -m src.export_artifacts
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.calibrate import fit_temperature
from src.conformal import calibrate_mondrian, choose_threshold, lac_scores, malignant_prob
from src.dataset import DX_LABELS
from src.error_analysis import MALIGNANT_CLASSES
from src.ood import accept_threshold_low_is_ood, fit_mahalanobis, mahalanobis_score, VAL_ACCEPT_RATE

CONFORMAL_ALPHA = 0.10  # Mondrian-LAC — see src/conformal.py's report for why this variant
MALIGNANT_IDX = [DX_LABELS.index(c) for c in MALIGNANT_CLASSES]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export frozen inference artifacts from the existing analysis caches.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--internal-cache-dir", type=str, default="reports/calibration/cache")
    parser.add_argument("--ood-cache-dir", type=str, default="reports/ood/cache")
    parser.add_argument("--output-json", type=str, default="models/inference_artifacts.json")
    parser.add_argument("--output-npz", type=str, default="models/mahalanobis_params.npz")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = sha256_of(checkpoint_path)
    print(f"[export] checkpoint: {checkpoint_path}")
    print(f"[export] sha256: {checkpoint_sha256}")

    internal_cache = Path(args.internal_cache_dir)
    ood_cache = Path(args.ood_cache_dir)

    # ---- Temperature: fit fresh on validation logits, exactly as src/calibrate.py does ----
    val_logits = torch.load(internal_cache / "val_logits.pt")
    val_labels = torch.load(internal_cache / "val_labels.pt")
    temperature = fit_temperature(val_logits, val_labels)
    print(f"[export] temperature (fit_temperature on val, n={len(val_labels)}) = {temperature!r}")

    val_labels_np = val_labels.numpy()
    val_probs = F.softmax(val_logits / temperature, dim=1).numpy()

    # ---- Mondrian-LAC alpha=0.10 per-class quantiles, calibrated on val only ----
    lac_all = lac_scores(val_probs)
    lac_true = lac_all[np.arange(len(val_labels_np)), val_labels_np]
    qhats = calibrate_mondrian(lac_true, val_labels_np, CONFORMAL_ALPHA, len(DX_LABELS))
    per_class_qhat = {DX_LABELS[i]: float(q) for i, q in enumerate(qhats)}
    print(f"[export] Mondrian-LAC alpha={CONFORMAL_ALPHA} per-class qhat: {per_class_qhat}")

    # ---- Malignant-sensitivity thresholds, chosen on val only ----
    val_mal_prob = malignant_prob(val_probs)
    val_mal_true = np.isin(val_labels_np, MALIGNANT_IDX).astype(int)
    thr90, achieved90 = choose_threshold(val_mal_prob, val_mal_true, 0.90)
    thr95, achieved95 = choose_threshold(val_mal_prob, val_mal_true, 0.95)
    print(f"[export] malignant threshold @90% target: {thr90:.6f} (achieved on val: {achieved90:.4f})")
    print(f"[export] malignant threshold @95% target: {thr95:.6f} (achieved on val: {achieved95:.4f})")

    # ---- Mahalanobis: refit means + Ledoit-Wolf precision on training features, ----
    # ---- threshold chosen on val features for 95% in-distribution acceptance ----
    train_features, _, train_labels = torch.load(ood_cache / "train.pt")
    means, precision = fit_mahalanobis(train_features, train_labels, len(DX_LABELS))
    val_features, _, _ = torch.load(ood_cache / "val.pt")
    val_maha = mahalanobis_score(val_features, means, precision)
    maha_threshold = accept_threshold_low_is_ood(val_maha, VAL_ACCEPT_RATE)
    maha_val_accept = float((val_maha <= maha_threshold).mean())
    print(f"[export] Mahalanobis threshold (Ledoit-Wolf covariance, {VAL_ACCEPT_RATE:.0%} val "
          f"acceptance target): {maha_threshold:.4f} (achieved on val: {maha_val_accept:.4f})")

    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, means=means, precision=precision)
    print(f"[export] wrote {npz_path} (means {means.shape}, precision {precision.shape})")

    artifacts = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "class_order": DX_LABELS,
        "temperature": temperature,
        "conformal": {
            "method": "mondrian_lac",
            "alpha": CONFORMAL_ALPHA,
            "calibrated_on": "internal_validation_only",
            "per_class_qhat": per_class_qhat,
        },
        "malignant_rule": {
            "classes": MALIGNANT_CLASSES,
            "calibrated_on": "internal_validation_only",
            "thresholds": {
                "sensitivity_90": {"target": 0.90, "threshold": thr90, "achieved_val_sensitivity": achieved90},
                "sensitivity_95": {"target": 0.95, "threshold": thr95, "achieved_val_sensitivity": achieved95},
            },
        },
        "ood_guard": {
            "method": "mahalanobis_ledoit_wolf",
            "calibrated_on": "internal_validation_only",
            "val_accept_rate_target": VAL_ACCEPT_RATE,
            "val_accept_rate_achieved": maha_val_accept,
            "threshold": maha_threshold,
            "means_precision_file": npz_path.name,
        },
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifacts, indent=2))
    print(f"[export] wrote {json_path}")


if __name__ == "__main__":
    main()
