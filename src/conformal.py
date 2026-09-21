"""
Conformal prediction sets and a malignant-first decision rule — analysis
only, read-only from cached logits (reports/calibration/cache/,
reports/isic2018_test/cache/). No re-inference, no retraining, and
TEMPERATURE stays fixed at app.py's current value (except the one explicit
val-reuse robustness check in Part A.4, which itself calls for refitting T
on half the validation set as the thing being measured).

Part A — split conformal prediction sets (Vovk et al.; LAC: Sadinle et al.
2019; APS, randomized: Romano, Sesia & Candès 2020), calibrated ONLY on the
internal validation set, evaluated on internal test and the external ISIC
2018 test set (both held out from calibration).

Part B — a malignant = {mel, bcc, akiec} vs benign screening rule: pick a
probability threshold on validation that hits a target sensitivity, then
apply that fixed threshold to internal test and external ISIC 2018.

Usage:
    python -m src.conformal
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.calibrate import fit_temperature
from src.dataset import DX_LABELS
from src.error_analysis import MALIGNANT_CLASSES
from src.evaluate_isic2018 import bootstrap_ci, fmt_ci

CURRENT_TEMPERATURE = 2.1235  # matches app.py; not refit here except Part A.4
N_BOOTSTRAP_DEFAULT = 1000
ALPHAS = (0.10, 0.05)
MALIGNANT_IDX = [DX_LABELS.index(c) for c in MALIGNANT_CLASSES]
MEL_IDX = DX_LABELS.index("mel")
NV_IDX = DX_LABELS.index("nv")

INTERNAL_CACHE = Path("reports/calibration/cache")
EXTERNAL_CACHE = Path("reports/isic2018_test/cache")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(cache_dir: Path, name: str):
    logits = torch.load(cache_dir / f"{name}_logits.pt")
    labels = torch.load(cache_dir / f"{name}_labels.pt")
    return logits, labels


def probs_at_T(logits: torch.Tensor, T: float = CURRENT_TEMPERATURE) -> np.ndarray:
    return F.softmax(logits / T, dim=1).numpy()


# ---------------------------------------------------------------------------
# Part A: nonconformity score functions
# ---------------------------------------------------------------------------

def lac_scores(probs: np.ndarray) -> np.ndarray:
    """(n, K) LAC nonconformity score for every candidate class: 1 - p(class | x)."""
    return 1.0 - probs


def aps_scores(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    (n, K) randomized APS nonconformity score for every candidate class
    (Romano, Sesia & Candès 2020): cumulative probability mass of all
    classes ranked strictly above it, plus a uniform-random share of its
    own probability mass (one U per row, reused across that row's classes
    so the tie-break is internally consistent).
    """
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum_incl = np.cumsum(sorted_probs, axis=1)
    cumsum_excl = cumsum_incl - sorted_probs
    u = rng.random((n, 1))
    sorted_scores = cumsum_excl + u * sorted_probs
    scores = np.empty_like(sorted_scores)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def score_all(name: str, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return lac_scores(probs) if name == "LAC" else aps_scores(probs, rng)


# ---------------------------------------------------------------------------
# Part A: split conformal calibration + prediction sets
# ---------------------------------------------------------------------------

def split_conformal_quantile(scores_true: np.ndarray, alpha: float) -> float:
    n = len(scores_true)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores_true, q_level, method="higher"))


def calibrate_marginal(scores_true: np.ndarray, alpha: float) -> float:
    return split_conformal_quantile(scores_true, alpha)


def calibrate_mondrian(scores_true: np.ndarray, cal_labels: np.ndarray, alpha: float, n_classes: int) -> np.ndarray:
    """Class-conditional (Mondrian) thresholds: one qhat per candidate class,
    calibrated only from calibration points whose TRUE label is that class."""
    qhats = np.full(n_classes, np.nan)
    for k in range(n_classes):
        mask = cal_labels == k
        if mask.sum() == 0:
            continue
        qhats[k] = split_conformal_quantile(scores_true[mask], alpha)
    return qhats


def prediction_sets(scores_all_classes: np.ndarray, qhat) -> np.ndarray:
    """qhat: scalar (marginal) or (K,) array (mondrian, per candidate class)."""
    return scores_all_classes <= qhat


def summarize_sets(inclusion: np.ndarray, true_labels: np.ndarray, target_coverage: float) -> dict:
    n = len(true_labels)
    hit = inclusion[np.arange(n), true_labels]
    set_sizes = inclusion.sum(axis=1)

    per_class = {}
    for ci, name in enumerate(DX_LABELS):
        mask = true_labels == ci
        support = int(mask.sum())
        if support == 0:
            per_class[name] = {"coverage": float("nan"), "avg_set_size": float("nan"), "support": 0, "below_target": False}
            continue
        cov = float(hit[mask].mean())
        per_class[name] = {
            "coverage": cov,
            "avg_set_size": float(set_sizes[mask].mean()),
            "support": support,
            "below_target": cov < target_coverage,
        }

    singleton_mask = set_sizes == 1
    frac_singleton = float(singleton_mask.mean())
    if singleton_mask.sum() > 0:
        singleton_class = inclusion[singleton_mask].argmax(axis=1)
        singleton_accuracy = float((singleton_class == true_labels[singleton_mask]).mean())
    else:
        singleton_accuracy = float("nan")

    mel_mask = true_labels == MEL_IDX
    mel_coverage = float(inclusion[mel_mask, MEL_IDX].mean()) if mel_mask.sum() > 0 else float("nan")

    return {
        "n": n,
        "marginal_coverage": float(hit.mean()),
        "avg_set_size": float(set_sizes.mean()),
        "per_class": per_class,
        "frac_singleton": frac_singleton,
        "singleton_accuracy": singleton_accuracy,
        "mel_coverage": mel_coverage,
    }


def run_variant(score_name: str, cond_type: str, alpha: float, val_probs: np.ndarray, val_labels: np.ndarray,
                 eval_sets: dict, rng: np.random.Generator):
    scores_all_val = score_all(score_name, val_probs, rng)
    scores_true_val = scores_all_val[np.arange(len(val_labels)), val_labels]

    if cond_type == "marginal":
        qhat = calibrate_marginal(scores_true_val, alpha)
    else:
        qhat = calibrate_mondrian(scores_true_val, val_labels, alpha, len(DX_LABELS))

    target = 1 - alpha
    results = {}
    for set_name, (probs, labels) in eval_sets.items():
        scores_all = score_all(score_name, probs, rng)
        inclusion = prediction_sets(scores_all, qhat)
        results[set_name] = summarize_sets(inclusion, labels, target)
    return qhat, results


def print_variant_report(score_name: str, cond_type: str, alpha: float, qhat, results: dict):
    target = 1 - alpha
    print(f"\n{'=' * 78}\n{score_name} / {cond_type} / alpha={alpha} (target coverage={target:.2f})")
    if cond_type == "marginal":
        print(f"  qhat = {qhat:.4f}")
    else:
        print(f"  qhat per class: {({DX_LABELS[i]: round(q, 4) for i, q in enumerate(qhat) if not np.isnan(q)})}")

    for set_name, s in results.items():
        print(f"\n  -- {set_name} (n={s['n']}) --")
        cov = s["marginal_coverage"]
        flag = "  <-- BELOW TARGET" if cov < target else ""
        print(f"     marginal coverage:        {cov:.4f} (target {target:.2f}){flag}")
        print(f"     avg prediction set size:  {s['avg_set_size']:.3f}")
        print(f"     singleton fraction:       {s['frac_singleton']:.4f}   accuracy among singletons: {s['singleton_accuracy']:.4f}")
        print(f"     P(mel in set | true mel): {s['mel_coverage']:.4f}")
        print("     per-class coverage (support, avg set size):")
        for cname in DX_LABELS:
            pc = s["per_class"][cname]
            if pc["support"] == 0:
                continue
            marker = "  <-- LOW" if pc["below_target"] else ""
            tag = " [MALIGNANT]" if cname in MALIGNANT_CLASSES else ""
            print(f"       {cname:8s}{tag:13s} cov={pc['coverage']:.4f}  n={pc['support']:4d}  avg_set_size={pc['avg_set_size']:.3f}{marker}")


# ---------------------------------------------------------------------------
# Part A.4: does reusing validation for both T-fitting and conformal
# calibration bias coverage?
# ---------------------------------------------------------------------------

def part_a4_split_robustness(val_logits, val_labels_t, test_logits, test_labels_t, rng, output_dir: Path):
    print("\n" + "#" * 78)
    print("# PART A.4 — reusing val for both temperature fitting and conformal calibration")
    print("#" * 78)

    n_val = len(val_labels_t)
    perm = np.random.default_rng(42).permutation(n_val)
    half = n_val // 2
    idx_a, idx_b = perm[:half], perm[half:]

    T_split = fit_temperature(val_logits[idx_a], val_labels_t[idx_a])
    print(f"T fit on val-half-A (n={len(idx_a)}): {T_split:.4f}  (vs. full-val fixed T={CURRENT_TEMPERATURE}, n={n_val})")

    full_val_probs = probs_at_T(val_logits)
    full_val_labels = val_labels_t.numpy()
    split_val_probs = F.softmax(val_logits[idx_b] / T_split, dim=1).numpy()
    split_val_labels = val_labels_t[idx_b].numpy()

    test_probs_full = probs_at_T(test_logits)
    test_probs_split = F.softmax(test_logits / T_split, dim=1).numpy()
    test_labels = test_labels_t.numpy()

    print(f"\n{'score':6s}{'alpha':>7s}{'full-val coverage':>20s}{'split-val coverage':>22s}{'delta':>10s}")
    rows = []
    for alpha in ALPHAS:
        for score_name in ("LAC", "APS"):
            scores_full = score_all(score_name, full_val_probs, rng)
            scores_true_full = scores_full[np.arange(len(full_val_labels)), full_val_labels]
            qhat_full = calibrate_marginal(scores_true_full, alpha)
            inc_full = prediction_sets(score_all(score_name, test_probs_full, rng), qhat_full)
            cov_full = float(inc_full[np.arange(len(test_labels)), test_labels].mean())

            scores_b = score_all(score_name, split_val_probs, rng)
            scores_true_b = scores_b[np.arange(len(split_val_labels)), split_val_labels]
            qhat_split = calibrate_marginal(scores_true_b, alpha)
            inc_split = prediction_sets(score_all(score_name, test_probs_split, rng), qhat_split)
            cov_split = float(inc_split[np.arange(len(test_labels)), test_labels].mean())

            delta = cov_split - cov_full
            print(f"{score_name:6s}{alpha:>7.2f}{cov_full:>20.4f}{cov_split:>22.4f}{delta:>+10.4f}")
            rows.append({"score": score_name, "alpha": alpha, "cov_full": cov_full, "cov_split": cov_split, "delta": delta})

    max_abs_delta = max(abs(r["delta"]) for r in rows)
    print(
        f"\nMax |delta| across score functions/alphas: {max_abs_delta:.4f}. "
        f"{'This is small relative to the target — reusing val for both purposes is not materially biasing internal-test coverage here.' if max_abs_delta < 0.02 else 'This is large enough to matter — treat coverage numbers calibrated on full val with some caution.'}"
    )
    print("(Marginal conformal only here — Mondrian on half-val would starve already-small minority classes further; that caveat applies on top of whatever this shows.)")
    return rows


# ---------------------------------------------------------------------------
# Part A driver + figures
# ---------------------------------------------------------------------------

def make_coverage_figure(figure_data: dict, output_dir: Path):
    set_names = ["internal_test", "external_isic2018"]
    variants = [("marginal", "LAC"), ("mondrian", "LAC"), ("marginal", "APS"), ("mondrian", "APS")]
    fig, axes = plt.subplots(len(ALPHAS), len(set_names), figsize=(15, 10), sharey=True)
    x = np.arange(len(DX_LABELS))
    width = 0.2

    for row, alpha in enumerate(ALPHAS):
        target = 1 - alpha
        for col, set_name in enumerate(set_names):
            ax = axes[row][col]
            for i, (cond_type, score_name) in enumerate(variants):
                s = figure_data[(alpha, score_name, cond_type)][set_name]
                covs = [s["per_class"][c]["coverage"] for c in DX_LABELS]
                ax.bar(x + (i - 1.5) * width, covs, width, label=f"{cond_type}-{score_name}")
            ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target {target:.2f}" if row == 0 and col == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(DX_LABELS, rotation=45, ha="right")
            ax.set_title(f"{set_name}, alpha={alpha}")
            ax.set_ylim(0, 1.05)
            if col == 0:
                ax.set_ylabel("per-class coverage")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Per-class conformal coverage: marginal vs. class-conditional (Mondrian), LAC vs. APS")
    fig.tight_layout()
    out_path = output_dir / "per_class_coverage.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[conformal] Saved {out_path}")


def make_set_size_figure(figure_data: dict, output_dir: Path):
    set_names = ["internal_test", "external_isic2018"]
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(15, 5.5), sharey=True)
    x = np.arange(len(DX_LABELS))
    width = 0.35

    for col, alpha in enumerate(ALPHAS):
        ax = axes[col]
        for i, set_name in enumerate(set_names):
            s = figure_data[(alpha, "APS", "mondrian")][set_name]
            sizes = [s["per_class"][c]["avg_set_size"] for c in DX_LABELS]
            ax.bar(x + (i - 0.5) * width, sizes, width, label=set_name)
        ax.set_xticks(x)
        ax.set_xticklabels(DX_LABELS, rotation=45, ha="right")
        ax.set_title(f"Mondrian-APS, alpha={alpha}")
        ax.set_ylabel("avg. prediction set size")
    axes[0].legend()
    fig.suptitle("Average prediction set size per true class (Mondrian-APS)")
    fig.tight_layout()
    out_path = output_dir / "set_size_by_class.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[conformal] Saved {out_path}")


def part_a(rng: np.random.Generator, output_dir: Path, n_boot: int, seed: int):
    val_logits, val_labels_t = load_split(INTERNAL_CACHE, "val")
    test_logits, test_labels_t = load_split(INTERNAL_CACHE, "test")
    ext_logits, ext_labels_t = load_split(EXTERNAL_CACHE, "isic2018_test")

    val_probs = probs_at_T(val_logits)
    val_labels = val_labels_t.numpy()
    eval_sets = {
        "internal_test": (probs_at_T(test_logits), test_labels_t.numpy()),
        "external_isic2018": (probs_at_T(ext_logits), ext_labels_t.numpy()),
    }

    print("=" * 78)
    print("PART A — split conformal prediction sets (calibrated on internal validation only)")
    print("=" * 78)
    val_counts = np.bincount(val_labels, minlength=len(DX_LABELS))
    print(f"Calibration set (internal val), n={len(val_labels)}, per-class counts: "
          f"{dict(zip(DX_LABELS, val_counts.tolist()))}")
    print("Note: Mondrian per-class thresholds for small classes (df, vasc, akiec) are fit on very "
          "few calibration points — expect noisier/more conservative per-class thresholds there.")

    figure_data = {}
    for alpha in ALPHAS:
        for score_name in ("LAC", "APS"):
            for cond_type in ("marginal", "mondrian"):
                qhat, results = run_variant(score_name, cond_type, alpha, val_probs, val_labels, eval_sets, rng)
                print_variant_report(score_name, cond_type, alpha, qhat, results)
                figure_data[(alpha, score_name, cond_type)] = results

    make_coverage_figure(figure_data, output_dir)
    make_set_size_figure(figure_data, output_dir)
    part_a4_split_robustness(val_logits, val_labels_t, test_logits, test_labels_t, rng, output_dir)


# ---------------------------------------------------------------------------
# Part B: malignant-first decision rule
# ---------------------------------------------------------------------------

def malignant_prob(probs: np.ndarray) -> np.ndarray:
    return probs[:, MALIGNANT_IDX].sum(axis=1)


def choose_threshold(mal_prob_val: np.ndarray, mal_true_val: np.ndarray, target_sens: float):
    """Largest threshold on validation malignant-probability that still hits
    >= target_sens sensitivity for the malignant class."""
    pos = mal_prob_val[mal_true_val == 1]
    pos_sorted = np.sort(pos)
    n_pos = len(pos_sorted)
    k = int(np.clip(np.ceil(target_sens * n_pos), 1, n_pos))
    threshold = float(pos_sorted[n_pos - k])
    achieved = float((pos >= threshold).mean())
    return threshold, achieved


def evaluate_rule(mal_prob: np.ndarray, mal_true: np.ndarray, threshold: float, n_boot: int, seed: int) -> dict:
    flagged = mal_prob >= threshold
    n = len(mal_true)

    def _sens(idx):
        pos = mal_true[idx] == 1
        return float(flagged[idx][pos].mean()) if pos.sum() > 0 else float("nan")

    def _spec(idx):
        neg = mal_true[idx] == 0
        return float((~flagged[idx][neg]).mean()) if neg.sum() > 0 else float("nan")

    def _ppv(idx):
        pred_pos = flagged[idx]
        return float((mal_true[idx][pred_pos] == 1).mean()) if pred_pos.sum() > 0 else float("nan")

    def _fp_per_tp(idx):
        tp = int((flagged[idx] & (mal_true[idx] == 1)).sum())
        fp = int((flagged[idx] & (mal_true[idx] == 0)).sum())
        return fp / tp if tp > 0 else float("nan")

    return {
        "n": n,
        "n_flagged": int(flagged.sum()),
        "sensitivity": bootstrap_ci(n, _sens, n_boot, seed),
        "specificity": bootstrap_ci(n, _spec, n_boot, seed),
        "ppv": bootstrap_ci(n, _ppv, n_boot, seed),
        "fp_per_tp": bootstrap_ci(n, _fp_per_tp, n_boot, seed),
    }


def make_malignant_rule_figure(all_results: dict, output_dir: Path, set_order: list):
    targets = (0.90, 0.95)
    fig, axes = plt.subplots(1, len(targets), figsize=(13, 5.5), sharey=True)
    metrics = ["sensitivity", "specificity", "ppv"]
    x = np.arange(len(set_order))
    width = 0.25

    for col, target in enumerate(targets):
        ax = axes[col]
        for i, metric in enumerate(metrics):
            vals = [all_results[(target, s)][metric][0] for s in set_order]
            los = [all_results[(target, s)][metric][0] - all_results[(target, s)][metric][1] for s in set_order]
            his = [all_results[(target, s)][metric][2] - all_results[(target, s)][metric][0] for s in set_order]
            ax.bar(x + (i - 1) * width, vals, width, yerr=[los, his], capsize=3, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(set_order, rotation=20, ha="right")
        ax.set_title(f"target malignant sensitivity = {target:.0%}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("value (95% bootstrap CI)")
    axes[0].legend()
    fig.suptitle("Malignant-first decision rule: sensitivity / specificity / PPV")
    fig.tight_layout()
    out_path = output_dir / "malignant_sensitivity_rule.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[conformal] Saved {out_path}")


def part_b(n_boot: int, seed: int, output_dir: Path):
    val_logits, val_labels_t = load_split(INTERNAL_CACHE, "val")
    test_logits, test_labels_t = load_split(INTERNAL_CACHE, "test")
    ext_logits, ext_labels_t = load_split(EXTERNAL_CACHE, "isic2018_test")

    def prep(logits, labels_t):
        probs = probs_at_T(logits)
        labels = labels_t.numpy()
        mal_true = np.isin(labels, MALIGNANT_IDX).astype(int)
        return probs, labels, mal_true, malignant_prob(probs)

    val_probs, val_labels, val_mal_true, val_mal_prob = prep(val_logits, val_labels_t)
    test_probs, test_labels, test_mal_true, test_mal_prob = prep(test_logits, test_labels_t)
    ext_probs, ext_labels, ext_mal_true, ext_mal_prob = prep(ext_logits, ext_labels_t)

    sets = {
        "internal_val (reference)": (val_probs, val_labels, val_mal_true, val_mal_prob),
        "internal_test": (test_probs, test_labels, test_mal_true, test_mal_prob),
        "external_isic2018": (ext_probs, ext_labels, ext_mal_true, ext_mal_prob),
    }

    print("\n" + "=" * 78)
    print("PART B — malignant-first decision rule (malignant = p(mel)+p(bcc)+p(akiec), T=2.1235)")
    print("Thresholds chosen on internal validation only; applied fixed to test/external.")
    print("=" * 78)

    all_results = {}
    for target in (0.90, 0.95):
        threshold, achieved_val_sens = choose_threshold(val_mal_prob, val_mal_true, target)
        n_mal_val = int(val_mal_true.sum())
        print(f"\nTarget malignant sensitivity {target:.0%}: threshold={threshold:.4f} "
              f"(achieved on val: {achieved_val_sens:.4f}, n_malignant_val={n_mal_val})")

        for set_name, (probs, labels, mal_true, mal_prob) in sets.items():
            r = evaluate_rule(mal_prob, mal_true, threshold, n_boot, seed)
            all_results[(target, set_name)] = r
            print(f"  {set_name:24s} n={r['n']:5d}  flagged={r['n_flagged']:5d}")
            print(f"      sensitivity: {fmt_ci(*r['sensitivity'])}   specificity: {fmt_ci(*r['specificity'])}   PPV: {fmt_ci(*r['ppv'])}")
            print(f"      benign lesions flagged per true malignant caught: {fmt_ci(*r['fp_per_tp'], digits=2)}")

            mel_mask = labels == MEL_IDX
            mel_recall_rule = float((mal_prob[mel_mask] >= threshold).mean()) if mel_mask.sum() > 0 else float("nan")
            argmax_pred = probs.argmax(axis=1)
            mel_called_nv_mask = mel_mask & (argmax_pred == NV_IDX)
            n_mel_called_nv = int(mel_called_nv_mask.sum())
            n_rescued = int((mal_prob[mel_called_nv_mask] >= threshold).sum()) if n_mel_called_nv > 0 else 0
            print(f"      melanoma recall under this rule: {mel_recall_rule:.4f} (n_mel={int(mel_mask.sum())}); "
                  f"of {n_mel_called_nv} true melanomas whose top-1 prediction was 'nv', "
                  f"{n_rescued} are now flagged malignant by this rule.")

    make_malignant_rule_figure(all_results, output_dir, list(sets.keys()))
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conformal prediction sets + malignant-first decision rule (analysis only, read-only cached logits).")
    parser.add_argument("--output-dir", type=str, default="reports/conformal")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    part_a(rng, output_dir, args.n_bootstrap, args.seed)
    part_b(args.n_bootstrap, args.seed, output_dir)


if __name__ == "__main__":
    main()
