"""
How much does the served v1 model's prediction depend on the metadata form?

For every internal-test image this computes the image features ONCE and then
runs the fusion head under three metadata variants:

- true:            the patient's real age / sex / site from the CSV
- form_defaults:   50 / male / back - what the old API and frontend filled in
                   whenever the user didn't touch the form
- all_unknown:     median age / "unknown" / "unknown" - what you get when a
                   caller has nothing to enter

and reports how often the top-1 class and the 95%-sensitivity malignant flag
change relative to the true metadata. If the flip rates are small, the
"fusion" in the project name is doing little; if they are large, silently
defaulted metadata was steering real predictions. Either answer is worth
knowing, and it needs no retraining.

Usage (same image paths as src/calibrate.py):
    python -m src.metadata_sensitivity \
        --images-dir-part1 /path/to/HAM10000_images_part_1 \
        --images-dir-part2 /path/to/HAM10000_images_part_2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader, Subset

from src.calibrate import build_datasets
from src.config import load_config
from src.dataset import DX_LABELS
from src.model import build_model
from src.utils import get_device, load_checkpoint

MEL = DX_LABELS.index("mel")


def encode(age, sex, loc, scaler, columns):
    row = dict.fromkeys(columns, 0.0)
    row["age_scaled"] = float(scaler.transform([[age]])[0][0])
    for col in (f"sex_{sex}", f"loc_{loc}"):
        if col in row:
            row[col] = 1.0
    return torch.tensor([[row[c] for c in columns]], dtype=torch.float32)


def head(model, feats, meta):
    fused = torch.cat((feats, model.metadata_processor(meta)), dim=1)
    return model.classifier_head(fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--artifacts", default="models/inference_artifacts.json")
    ap.add_argument("--metadata-csv", default="data/HAM10000_metadata.csv")
    ap.add_argument("--images-dir-part1", required=True)
    ap.add_argument("--images-dir-part2", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N test images (debugging)")
    ap.add_argument("--out", default="reports/metadata_sensitivity/report.json")
    args = ap.parse_args()

    config = load_config(args.config)
    artifacts = json.loads(Path(args.artifacts).read_text())
    T = artifacts["temperature"]
    mal_idx = [DX_LABELS.index(c) for c in artifacts["malignant_rule"]["classes"]]
    mal_thr = artifacts["malignant_rule"]["thresholds"]["sensitivity_95"]["threshold"]

    _, test_ds, use_metadata = build_datasets(config, args.metadata_csv, args.images_dir_part1, args.images_dir_part2)
    if not use_metadata:
        raise SystemExit("config has use_metadata=false; nothing to measure")
    device = get_device(config["train"]["device"])
    model = build_model(config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device).eval()

    cols = test_ds.metadata_columns
    variants = {
        "form_defaults": encode(50.0, "male", "back", test_ds.age_scaler, cols).to(device),
        "all_unknown": encode(test_ds.age_median, "unknown", "unknown", test_ds.age_scaler, cols).to(device),
    }
    ds = Subset(test_ds, range(min(args.limit, len(test_ds)))) if args.limit else test_ds
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    logits = {k: [] for k in ["true", *variants]}
    labels = []
    with torch.no_grad():
        for images, metas, y in loader:
            feats = model.forward_features(images.to(device))
            logits["true"].append(head(model, feats, metas.to(device)).cpu())
            for name, v in variants.items():
                logits[name].append(head(model, feats, v.expand(len(feats), -1)).cpu())
            labels.append(y)
    y = torch.cat(labels).numpy()
    malignant_true = np.isin(y, mal_idx)

    report = {"n": int(len(y)), "malignant_threshold": mal_thr, "variants": {}}
    base_pred = base_flag = None
    for name, chunks in logits.items():
        lg = torch.cat(chunks)
        probs = torch.softmax(lg / T, 1).numpy()
        pred = probs.argmax(1)
        flag = probs[:, mal_idx].sum(1) >= mal_thr
        r = {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mel_recall": float(recall_score(y, pred, labels=[MEL], average=None, zero_division=0)[0]),
            "malignant_sensitivity": float(flag[malignant_true].mean()) if malignant_true.any() else None,
            "malignant_flag_rate": float(flag.mean()),
        }
        if name == "true":
            base_pred, base_flag = pred, flag
        else:
            r["top1_changed_vs_true"] = float((pred != base_pred).mean())
            r["malignant_flag_changed_vs_true"] = float((flag != base_flag).mean())
            lost = base_flag & ~flag & malignant_true
            r["true_malignant_lost_flag"] = int(lost.sum())
        report["variants"][name] = r
        print(f"[{name:13s}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
