"""
Reproduces every v2 number in the README from the committed caches in
reports/v2/ and reports/{calibration,isic2018_test}/cache/ (v1). No model,
no images, no GPU: it only reads logits, so anyone can check the claims.

    python -m src.analyze_v2            # prints markdown tables
    python -m src.analyze_v2 --json reports/v2/analysis.json

What it reports, v1 vs v2 on the same images:
1. Headline metrics (internal HAM10000 test, external ISIC 2018 test)
2. Fusion vs image-only, paired: does metadata earn its place?
3. Metadata at inference: the fusion model with vs without its form fields
4. The served safety layer re-derived for v2 from validation only:
   temperature, 95%-sensitivity malignant threshold, Mondrian-LAC sets
   (alpha 0.10), then scored on internal test and ISIC 2018.

Runs are loaded by name. fusion_seed1 is reported separately because its
training was interrupted (no history/summary/val cache was written), so its
checkpoint is the best of an unknown number of early epochs, not a finished
replicate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.calibrate import compute_ece_mce, fit_temperature
from src.conformal import calibrate_mondrian, choose_threshold, lac_scores, malignant_prob
from src.dataset import DX_LABELS
from src.evaluate_v2 import METRICS
from src.v2.engine import MALIGNANT_IDX, paired_bootstrap

V2 = Path("reports/v2")
V1_INTERNAL = Path("reports/calibration/cache")
V1_EXTERNAL = Path("reports/isic2018_test/cache")
ALPHA = 0.10
N_BOOT = 2000


def _load(path: Path) -> np.ndarray:
    return torch.load(path).numpy()


def eval_cache(run: str, split: str, scenario: str):
    """split in {internal_test, external_isic2018}. Image-only runs were
    evaluated before the scenario rename, so accept either label."""
    base = V2 / run / "eval" / "cache"
    for name in ([scenario] if scenario != "image_only_model" else ["image_only_model", "with_metadata"]):
        lp = base / f"{split}_{name}_logits.pt"
        if lp.exists():
            return _load(lp), _load(base / f"{split}_{name}_labels.pt")
    return None


def v1_cache(split: str):
    if split == "internal_test":
        return _load(V1_INTERNAL / "test_logits.pt"), _load(V1_INTERNAL / "test_labels.pt")
    return _load(V1_EXTERNAL / "isic2018_test_logits.pt"), _load(V1_EXTERNAL / "isic2018_test_labels.pt")


def v1_val():
    return _load(V1_INTERNAL / "val_logits.pt"), _load(V1_INTERNAL / "val_labels.pt")


def v2_val(run: str):
    p = V2 / run / "cache" / "val_logits.pt"
    return (_load(p), _load(V2 / run / "cache" / "val_labels.pt")) if p.exists() else None


def fmt(v, ci=None, signed=False):
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def compare(y, a, b, metrics=("balanced_accuracy", "macro_f1", "mel_recall", "mel_auc", "malignant_auc")):
    out = {}
    for m in metrics:
        delta, ci = paired_bootstrap(METRICS[m], y, a, b, n=N_BOOT)
        out[m] = {"a": float(METRICS[m](y, a)), "b": float(METRICS[m](y, b)), "delta": delta, "ci95": ci}
    return out


def safety_layer(val, sets: dict) -> dict:
    """Everything fitted on `val` only, then applied unchanged to each set."""
    val_logits, val_y = val
    T = fit_temperature(torch.from_numpy(val_logits), torch.from_numpy(val_y))
    vp = F.softmax(torch.from_numpy(val_logits) / T, 1).numpy()
    thr, _ = choose_threshold(malignant_prob(vp), np.isin(val_y, MALIGNANT_IDX).astype(int), 0.95)
    lac = lac_scores(vp)
    qhat = calibrate_mondrian(lac[np.arange(len(val_y)), val_y], val_y, ALPHA, len(DX_LABELS))
    out = {"temperature": T, "malignant_threshold": thr, "sets": {}}
    rng = np.random.default_rng(0)
    for name, (logits, y) in sets.items():
        p = F.softmax(torch.from_numpy(logits) / T, 1).numpy()
        mal_true = np.isin(y, MALIGNANT_IDX)
        flagged = malignant_prob(p) >= thr
        inclusion = lac_scores(p) <= qhat
        sens, spec = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            sens.append(flagged[i][mal_true[i]].mean())
            spec.append((~flagged[i][~mal_true[i]]).mean())
        ece, _, _ = compute_ece_mce(p.max(1), (p.argmax(1) == y).astype(float))
        out["sets"][name] = {
            "n": int(len(y)),
            "malignant_sensitivity": float(flagged[mal_true].mean()),
            "malignant_sensitivity_ci": np.percentile(sens, [2.5, 97.5]).tolist(),
            "malignant_specificity": float((~flagged[~mal_true]).mean()),
            "malignant_specificity_ci": np.percentile(spec, [2.5, 97.5]).tolist(),
            "benign_flagged_per_malignant_caught": float((flagged & ~mal_true).sum() / max(1, (flagged & mal_true).sum())),
            "conformal_coverage": float(inclusion[np.arange(len(y)), y].mean()),
            "conformal_mean_set_size": float(inclusion.sum(1).mean()),
            "ece": float(ece),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write all numbers to this path")
    args = ap.parse_args()
    report = {}
    splits = ["internal_test", "external_isic2018"]

    # 1. headline
    print("## v1 vs v2 (paired bootstrap, 95% CI)\n")
    report["v1_vs_v2"] = {}
    for run, scenario in [("fusion", "with_metadata"), ("fusion", "metadata_withheld"), ("image_only", "image_only_model"),
                          ("fusion_seed1", "with_metadata")]:
        for split in splits:
            c = eval_cache(run, split, scenario)
            if c is None:
                continue
            v1_logits, v1_y = v1_cache(split)
            assert np.array_equal(v1_y, c[1]), f"{run}/{split}: labels not aligned with v1"
            r = compare(c[1], v1_logits, c[0])
            report["v1_vs_v2"][f"{run}|{scenario}|{split}"] = r
            print(f"| {split} | {run} ({scenario}) | " + " | ".join(
                f"{m}: {v['a']:.3f} -> {v['b']:.3f} {fmt(v['delta'], v['ci95'], True)}" for m, v in r.items()) + " |")

    # 2. fusion vs image-only
    print("\n## fusion (with metadata) vs image-only, paired\n")
    report["fusion_vs_image_only"] = {}
    for split in splits:
        img, fus = eval_cache("image_only", split, "image_only_model"), eval_cache("fusion", split, "with_metadata")
        assert np.array_equal(img[1], fus[1])
        r = compare(fus[1], img[0], fus[0], ("balanced_accuracy", "macro_f1", "mel_auc", "malignant_auc"))
        report["fusion_vs_image_only"][split] = r
        print(f"| {split} | " + " | ".join(f"{m}: {v['a']:.3f} -> {v['b']:.3f} {fmt(v['delta'], v['ci95'], True)}" for m, v in r.items()) + " |")

    # 3. metadata at inference
    print("\n## fusion model: metadata withheld -> provided, paired\n")
    report["metadata_at_inference"] = {}
    for split in splits:
        wo, w = eval_cache("fusion", split, "metadata_withheld"), eval_cache("fusion", split, "with_metadata")
        r = compare(w[1], wo[0], w[0], ("balanced_accuracy", "macro_f1", "mel_auc", "malignant_auc"))
        report["metadata_at_inference"][split] = r
        print(f"| {split} | " + " | ".join(f"{m}: {v['a']:.3f} -> {v['b']:.3f} {fmt(v['delta'], v['ci95'], True)}" for m, v in r.items()) + " |")

    # 4. safety layer re-derived from validation only
    print("\n## safety layer (T, malignant threshold, conformal sets) fitted on validation only\n")
    report["safety_layer"] = {
        "v1": safety_layer(v1_val(), {s: v1_cache(s) for s in splits}),
        "v2_fusion": safety_layer(v2_val("fusion"), {s: eval_cache("fusion", s, "with_metadata") for s in splits}),
        "v2_image_only": safety_layer(v2_val("image_only"), {s: eval_cache("image_only", s, "image_only_model") for s in splits}),
    }
    for model, r in report["safety_layer"].items():
        print(f"{model}: T={r['temperature']:.3f} malignant threshold={r['malignant_threshold']:.4f}")
        for split, s in r["sets"].items():
            print(f"  {split}: malignant sens {s['malignant_sensitivity']:.3f} {s['malignant_sensitivity_ci']}, "
                  f"spec {s['malignant_specificity']:.3f} {s['malignant_specificity_ci']}, "
                  f"benign flagged per malignant caught {s['benign_flagged_per_malignant_caught']:.2f}, "
                  f"conformal coverage {s['conformal_coverage']:.3f} (target {1 - ALPHA:.2f}), "
                  f"mean set size {s['conformal_mean_set_size']:.2f}, ECE {s['ece']:.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
