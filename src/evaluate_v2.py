"""
Score a v2 checkpoint against v1 on the SAME images, with paired bootstrap CIs.

    python -m src.evaluate_v2 --checkpoint runs_v2/fusion/best_model_v2.pt \
        --isic-images-dir /path/to/ISIC2018_Task3_Test_Images \
        --isic-groundtruth /path/to/ISIC2018_Task3_Test_GroundTruth.tab

Reports three scenarios per test set:
- with metadata      (what the fusion model sees when the form is filled in)
- metadata withheld  (every field missing: what a user who skips the form gets)
- v1 baseline        (cached logits in reports/, same images, same order)

and the paired delta v2 - v1 with a 95% bootstrap CI. A gain whose CI
crosses zero is not a gain yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import DX_LABELS, load_metadata, stratified_split
from src.utils import get_device
from src.v2.data import LesionDataset, resolve_image_paths
from src.v2.engine import MALIGNANT_IDX, MEL, apply_overrides, paired_bootstrap, predict_logits, summarize
from src.v2.metadata import MetadataEncoder
from src.v2.model import build_fusion_model
from src.v2.transforms import get_eval_transforms


def _probs(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits), 1).numpy()


METRICS = {
    "balanced_accuracy": lambda y, lg: balanced_accuracy_score(y, lg.argmax(1)),
    "macro_f1": lambda y, lg: f1_score(y, lg.argmax(1), average="macro", zero_division=0),
    "mel_recall": lambda y, lg: recall_score(y, lg.argmax(1), labels=[MEL], average=None, zero_division=0)[0],
    "mel_auc": lambda y, lg: roc_auc_score(y == MEL, _probs(lg)[:, MEL]),
    "malignant_auc": lambda y, lg: roc_auc_score(np.isin(y, MALIGNANT_IDX), _probs(lg)[:, MALIGNANT_IDX].sum(1)),
}


def load_v2(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("format") != "cancer-fusion-v2":
        raise ValueError(f"{checkpoint} is not a v2 checkpoint (format={state.get('format')})")
    if state["class_order"] != DX_LABELS:
        raise ValueError(f"class order mismatch: {state['class_order']} vs {DX_LABELS}")
    encoder = MetadataEncoder.from_dict(state["metadata_encoder"]) if state["use_metadata"] else None
    model = build_fusion_model(state["model_config"], state["use_metadata"], encoder.dim if encoder else 0, pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), encoder, state


def compare_with_v1(name: str, y: np.ndarray, v2_logits: np.ndarray, v1_cache: Path, v1_prefix: str, n_boot: int) -> dict | None:
    lp, yp = v1_cache / f"{v1_prefix}_logits.pt", v1_cache / f"{v1_prefix}_labels.pt"
    if not lp.exists():
        print(f"[evaluate_v2] no v1 cache at {lp}; skipping paired comparison for {name}")
        return None
    v1_logits, v1_labels = torch.load(lp).numpy(), torch.load(yp).numpy()
    if len(v1_labels) != len(y) or not np.array_equal(v1_labels, y):
        raise RuntimeError(
            f"{name}: v1 cached labels do not line up with this evaluation's images "
            f"(n={len(v1_labels)} vs {len(y)}). Paired comparison would be meaningless; "
            f"check that the metadata CSV / ground-truth file and image folders match what v1 used."
        )
    report = {}
    for metric, fn in METRICS.items():
        try:
            delta, ci = paired_bootstrap(fn, y, v1_logits, v2_logits, n=n_boot)
        except ValueError as e:  # e.g. an AUC on a set with no positives
            print(f"[evaluate_v2] {name}: skipping {metric} ({e})")
            continue
        report[metric] = {"v1": float(fn(y, v1_logits)), "v2": float(fn(y, v2_logits)), "delta": delta, "ci95": ci}
    return report


def run_set(name, df, model, encoder, state, device, tta, bs, workers, amp):
    tf = get_eval_transforms(state["image_size"], state["color_constancy"])
    results = {}
    scenarios = [("with_metadata", False)] + ([("metadata_withheld", True)] if encoder is not None else [])
    for scenario, drop_all in scenarios:
        ds = LesionDataset(df, tf, encoder, drop_all_metadata=drop_all)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        logits, labels = predict_logits(model, loader, device, tta_views=tta, amp=amp)
        results[scenario] = (logits.numpy(), labels.numpy())
        m = summarize(logits, labels)
        print(f"[{name} | {scenario}] n={len(labels)} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config_v2.yaml", help="only data paths/split are read from it")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--isic-images-dir")
    ap.add_argument("--isic-groundtruth")
    ap.add_argument("--v1-internal-cache", default="reports/calibration/cache")
    ap.add_argument("--v1-external-cache", default="reports/isic2018_test/cache")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    device = get_device(config["train"].get("device", "cuda"))
    amp = device.type == "cuda"
    model, encoder, state = load_v2(args.checkpoint, device)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / "eval")
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": args.checkpoint, "tta_views": args.tta, "sets": {}}

    d = config["data"]
    _, _, test_df = stratified_split(load_metadata(d["metadata_csv"]), d["train_split"], d["val_split"], d["test_split"], d["random_seed"])
    sets = [("internal_test", resolve_image_paths(test_df, [d["images_dir_part1"], d["images_dir_part2"]], d.get("image_extension", ".jpg")),
             Path(args.v1_internal_cache), "test")]
    if args.isic_images_dir and args.isic_groundtruth:
        from src.evaluate_isic2018 import load_isic_groundtruth
        isic_df = load_isic_groundtruth(args.isic_groundtruth)
        sets.append(("external_isic2018", resolve_image_paths(isic_df, [args.isic_images_dir]),
                     Path(args.v1_external_cache), "isic2018_test"))

    for name, df, v1_cache, v1_prefix in sets:
        results = run_set(name, df, model, encoder, state, device, args.tta, args.batch_size, args.num_workers, amp)
        entry = {}
        for scenario, (logits, labels) in results.items():
            torch.save(torch.from_numpy(logits), out_dir / "cache" / f"{name}_{scenario}_logits.pt")
            torch.save(torch.from_numpy(labels), out_dir / "cache" / f"{name}_{scenario}_labels.pt")
            entry[scenario] = {"metrics": summarize(torch.from_numpy(logits), torch.from_numpy(labels)),
                               "vs_v1": compare_with_v1(name, labels, logits, v1_cache, v1_prefix, args.n_boot)}
        report["sets"][name] = entry

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("\n| set | scenario | metric | v1 | v2 | delta [95% CI] |\n|---|---|---|---|---|---|")
    for name, entry in report["sets"].items():
        for scenario, r in entry.items():
            for metric, v in (r["vs_v1"] or {}).items():
                print(f"| {name} | {scenario} | {metric} | {v['v1']:.4f} | {v['v2']:.4f} | "
                      f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    print(f"\n[evaluate_v2] wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
