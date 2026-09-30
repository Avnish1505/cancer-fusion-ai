"""
v2 training entry point.

    python -m src.train_v2 --config configs/config_v2.yaml
    # ablations, everything else identical:
    python -m src.train_v2 --config configs/config_v2.yaml --set train.use_metadata=false --set run_name=image_only
    python -m src.train_v2 --config configs/config_v2.yaml --set train.seed=1 --set run_name=fusion_seed1

What changed vs src/train.py (v1), each item aimed at a measured v1 weakness:
- timm backbone (default EfficientNetV2-S, ImageNet-21k pretrained) at 384px
  instead of ResNet50 at 224px squashed.
- sqrt-inverse-frequency class weights instead of full inverse frequency.
- AdamW with separate LRs for pretrained backbone vs new head, 1-epoch
  warmup + cosine decay (v1 had a constant LR), AMP, gradient clipping.
- EMA of weights; model selection on EMA weights by val balanced accuracy
  (the ISIC 2018 Task 3 metric), with melanoma recall logged every epoch.
- Metadata dropout + explicit missing flags (see src/v2/metadata.py).
- Final val/test logits written with 8-view TTA in the same cache format
  v1 uses, so the existing calibrate/conformal/export tooling can read them.

The split is v1's split (same function, CSV and seed), so v1 and v2 are
scored on the same test images.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import DX_LABELS, load_metadata, stratified_split
from src.utils import get_device
from src.v2.data import LesionDataset, class_weights, resolve_image_paths
from src.v2.engine import ModelEma, apply_overrides, predict_logits, summarize, warmup_cosine
from src.v2.metadata import MetadataEncoder
from src.v2.model import build_fusion_model
from src.v2.transforms import get_eval_transforms, get_train_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed over bit-exactness: v2 runs are compared across seeds, not bit-for-bit.
    torch.backends.cudnn.benchmark = True


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_split(config: dict):
    d = config["data"]
    df = load_metadata(d["metadata_csv"])
    train_df, val_df, test_df = stratified_split(df, d["train_split"], d["val_split"], d["test_split"], d["random_seed"])
    image_dirs = [d["images_dir_part1"], d["images_dir_part2"]]
    ext = d.get("image_extension", ".jpg")
    return tuple(resolve_image_paths(x, image_dirs, ext) for x in (train_df, val_df, test_df))


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, device: torch.device, drop_last: bool = False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=drop_last,
                      persistent_workers=workers > 0)


def main(config: dict) -> Path:
    t = config["train"]
    run_name = config.get("run_name") or ("fusion" if t.get("use_metadata", True) else "image_only")
    out_dir = Path(config["paths"]["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(t.get("seed", 0)))
    device = get_device(t.get("device", "cuda"))
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    print(f"[train_v2] run={run_name} device={device} amp={amp} out={out_dir}")

    train_df, val_df, test_df = build_split(config)
    use_metadata = bool(t.get("use_metadata", True))
    encoder = MetadataEncoder.fit(train_df) if use_metadata else None
    size = int(t["image_size"])
    cc = bool(t.get("color_constancy", True))

    train_ds = LesionDataset(train_df, get_train_transforms(size, cc), encoder, meta_dropout=float(t.get("meta_dropout", 0.25)))
    val_ds = LesionDataset(val_df, get_eval_transforms(size, cc), encoder)
    test_ds = LesionDataset(test_df, get_eval_transforms(size, cc), encoder)
    workers = int(t.get("num_workers", 4))
    bs = int(t["batch_size"])
    train_loader = make_loader(train_ds, bs, True, workers, device, drop_last=True)
    val_loader = make_loader(val_ds, bs * 2, False, workers, device)
    test_loader = make_loader(test_ds, bs * 2, False, workers, device)
    print(f"[train_v2] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"metadata_dim={encoder.dim if encoder else 0}")

    model = build_fusion_model(config["model"], use_metadata, encoder.dim if encoder else 0).to(device)
    if t.get("channels_last", True) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    weights = class_weights(train_ds.labels, float(t.get("class_weight_power", 0.5))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(t.get("label_smoothing", 0.0)))
    print(f"[train_v2] class weights: {dict(zip(DX_LABELS, weights.cpu().numpy().round(3).tolist()))}")

    optimizer = torch.optim.AdamW(
        model.param_groups(float(t["learning_rate"]), float(t.get("head_lr_mult", 10.0)), float(t.get("weight_decay", 0.02)))
    )
    epochs = int(t["num_epochs"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(epochs * steps_per_epoch, int(t.get("warmup_epochs", 1)) * steps_per_epoch)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEma(model, float(t.get("ema_decay", 0.998))) if t.get("ema", True) else None

    select = t.get("select_metric", "balanced_accuracy")
    patience = int(t.get("early_stopping_patience", 8))
    best, best_epoch, history = -1.0, 0, []
    ckpt_path = out_dir / "best_model_v2.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for images, meta, labels in train_loader:
            images = images.to(device, non_blocking=True)
            if t.get("channels_last", True) and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            meta = meta.to(device, non_blocking=True) if meta.numel() else None
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(images, meta), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running += loss.item() * images.size(0)
            seen += images.size(0)

        eval_model = ema.module if ema is not None else model
        val_logits, val_labels = predict_logits(eval_model, val_loader, device, tta_views=1, amp=amp)
        m = summarize(val_logits, val_labels)
        m.update(epoch=epoch, train_loss=running / max(1, seen), seconds=time.time() - t0)
        history.append(m)
        print(f"[epoch {epoch}/{epochs}] loss={m['train_loss']:.4f} val_bacc={m['balanced_accuracy']:.4f} "
              f"val_mF1={m['macro_f1']:.4f} val_mel_rec={m['mel_recall']:.4f} "
              f"val_mel_auc={m.get('mel_auc', float('nan')):.4f} ({m['seconds']:.0f}s)")

        if m[select] > best:
            best, best_epoch = m[select], epoch
            torch.save({
                "format": "cancer-fusion-v2",
                "model_state_dict": eval_model.state_dict(),
                "model_config": config["model"],
                "use_metadata": use_metadata,
                "metadata_encoder": encoder.to_dict() if encoder else None,
                "image_size": size,
                "color_constancy": cc,
                "class_order": DX_LABELS,
                "epoch": epoch,
                "val_metrics": m,
                "select_metric": select,
                "config": config,
                "git_commit": git_commit(),
            }, ckpt_path)
            print(f"[train_v2] new best {select}={best:.4f} -> {ckpt_path}")
        elif epoch - best_epoch >= patience:
            print(f"[train_v2] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---- final: best checkpoint, TTA logits on val + test, v1-compatible cache ----
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_model = build_fusion_model(config["model"], use_metadata, encoder.dim if encoder else 0, pretrained=False).to(device)
    best_model.load_state_dict(state["model_state_dict"])
    cache = out_dir / "cache"
    cache.mkdir(exist_ok=True)
    tta = int(t.get("tta_views", 8))
    summary = {"run": run_name, "best_epoch": best_epoch, "tta_views": tta}
    for split_name, loader, df in (("val", val_loader, val_df), ("test", test_loader, test_df)):
        logits, labels = predict_logits(best_model, loader, device, tta_views=tta, amp=amp)
        torch.save(logits, cache / f"{split_name}_logits.pt")
        torch.save(labels, cache / f"{split_name}_labels.pt")
        (cache / f"{split_name}_image_ids.json").write_text(json.dumps(df["image_id"].tolist()))
        summary[split_name] = summarize(logits, labels)
        print(f"[train_v2] {split_name} (TTA x{tta}): " + " ".join(f"{k}={v:.4f}" for k, v in summary[split_name].items()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the v2 fusion model")
    parser.add_argument("--config", default="configs/config_v2.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config value, e.g. --set train.use_metadata=false")
    args = parser.parse_args()
    main(apply_overrides(load_config(args.config), args.overrides))
