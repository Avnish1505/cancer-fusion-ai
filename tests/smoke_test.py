"""
Smoke test: builds a tiny FAKE HAM10000-like dataset (random images + metadata)
and runs the entire pipeline end-to-end (dataset loading, splitting, model,
one training epoch, checkpoint save/load, evaluate) to catch integration bugs
BEFORE running on the real dataset on Kaggle where a crash wastes GPU quota.

This does not test model accuracy (data is random noise) — only that the
code runs correctly without crashing, with correct tensor shapes.
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import load_metadata, stratified_split, HAM10000Dataset, compute_class_weights, DX_LABELS
from src.transforms import get_train_transforms, get_eval_transforms
from src.model import build_model
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint, EarlyStopping

TEST_DIR = Path("/tmp/ham10000_smoke_test")


def build_fake_dataset(n_samples: int = 60):
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    img_dir1 = TEST_DIR / "part1"
    img_dir2 = TEST_DIR / "part2"
    img_dir1.mkdir(parents=True)
    img_dir2.mkdir(parents=True)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_samples):
        image_id = f"ISIC_{i:07d}"
        label = DX_LABELS[i % len(DX_LABELS)]
        lesion_id = f"HAM_{i // 2:04d}"  # simulate some lesions having 2 photos

        # Randomly place image in part1 or part2, like the real dataset
        target_dir = img_dir1 if i % 2 == 0 else img_dir2
        img = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8))
        img.save(target_dir / f"{image_id}.jpg")

        rows.append({
            "image_id": image_id, "dx": label, "lesion_id": lesion_id,
            "age": rng.integers(20, 80),
            "sex": rng.choice(["male", "female"]),
            "localization": rng.choice(["back", "face", "chest"])
        })

    df = pd.DataFrame(rows)
    csv_path = TEST_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False)

    return str(csv_path), str(img_dir1), str(img_dir2)


def build_fake_config(csv_path, img_dir1, img_dir2):
    config = {
        "data": {
            "metadata_csv": csv_path,
            "images_dir_part1": img_dir1,
            "images_dir_part2": img_dir2,
            "image_extension": ".jpg",
            "train_split": 0.6,
            "val_split": 0.2,
            "test_split": 0.2,
            "random_seed": 42,
        },
        "model": {
            "backbone": "resnet50",
            "pretrained": False,  # no internet download needed for smoke test
            "num_classes": 7,
            "dropout": 0.3,
        },
        "train": {
            "image_size": 64,
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 5,
            "use_class_weights": True,
            "use_metadata": True, # ENABLED FOR FUSION TEST
            "num_workers": 0,
            "device": "cpu",
        },
        "paths": {
            "checkpoint_dir": str(TEST_DIR / "checkpoints"),
            "log_dir": str(TEST_DIR / "logs"),
            "results_dir": str(TEST_DIR / "results"),
        },
        "logging": {"log_every_n_steps": 1},
    }
    config_path = TEST_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def run_smoke_test():
    print("=" * 60)
    print("SMOKE TEST: Phase 1 pipeline (fake data)")
    print("=" * 60)

    print("\n[1/7] Building fake dataset...")
    csv_path, img_dir1, img_dir2 = build_fake_dataset(n_samples=60)
    print(f"  Created 60 fake images across 2 folders + metadata CSV")

    print("\n[2/7] Building fake config...")
    config_path = build_fake_config(csv_path, img_dir1, img_dir2)
    config = load_config(config_path)
    print("  Config loaded and validated OK")

    print("\n[3/7] Testing metadata loading + stratified split...")
    df = load_metadata(config["data"]["metadata_csv"])
    assert len(df) == 60, f"Expected 60 rows, got {len(df)}"
    train_df, val_df, test_df = stratified_split(
        df, 0.6, 0.2, 0.2, random_seed=42
    )
    assert len(train_df) + len(val_df) + len(test_df) == 60, "Split sizes don't sum to total"
    # Critical check: no lesion_id should appear in more than one split
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert not (train_lesions & val_lesions), "LEAKAGE: lesion_id shared between train/val!"
    assert not (train_lesions & test_lesions), "LEAKAGE: lesion_id shared between train/test!"
    assert not (val_lesions & test_lesions), "LEAKAGE: lesion_id shared between val/test!"
    print(f"  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("  No lesion-level data leakage between splits (verified)")

    print("\n[4/7] Testing Dataset + DataLoader...")
    import torch
    from torch.utils.data import DataLoader

    train_dataset = HAM10000Dataset(
        train_df, [img_dir1, img_dir2], ".jpg",
        transform=get_train_transforms(64),
        use_metadata=config["train"]["use_metadata"]
    )
    val_dataset = HAM10000Dataset(
        val_df, [img_dir1, img_dir2], ".jpg",
        transform=get_eval_transforms(64),
        use_metadata=config["train"]["use_metadata"],
        age_scaler=train_dataset.age_scaler,
        metadata_columns=train_dataset.metadata_columns
    )
    assert len(train_dataset) == len(train_df), "Dataset length mismatch"

    img, meta, label = train_dataset[0]
    assert img.shape == (3, 64, 64), f"Expected image shape (3,64,64), got {img.shape}"
    assert isinstance(label.item(), int), "Label should be an integer tensor"
    print(f"  Train sample shapes: img={img.shape}, meta={meta.shape}, label={label.shape}")

    # --- CRITICAL CHECK: Verify metadata columns are aligned across splits ---
    val_img, val_meta, val_label = val_dataset[0]
    assert meta.shape == val_meta.shape, \
        f"FATAL: Metadata shape mismatch! train={meta.shape}, val={val_meta.shape}"
    print(f"  Metadata shape aligned across splits (dim={meta.shape[0]}), verified.")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    batch_images, batch_metas, batch_labels = next(iter(train_loader))
    assert batch_images.shape[0] <= 4, "Batch size exceeded"
    print(f"  Batch shapes: images={batch_images.shape}, metas={batch_metas.shape}")

    print("\n[5/7] Testing class weight computation...")
    weights = compute_class_weights(train_df)
    assert weights.shape[0] == 7, f"Expected 7 class weights, got {weights.shape[0]}"
    assert not torch.isnan(weights).any(), "NaN in class weights!"
    print(f"  Class weights: {weights.numpy().round(3)}")

    print("\n[6/7] Testing model forward pass + one training step...")
    model = build_model(config)
    device = get_device("cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    model.train()
    batch_images = batch_images.to(device)
    batch_metas = batch_metas.to(device)
    batch_labels = batch_labels.to(device)

    # This will fail until you implement the fusion model in model.py
    outputs = model(batch_images, batch_metas)
    assert outputs.shape == (batch_images.shape[0], 7), f"Expected logits shape (B,7), got {outputs.shape}"

    loss = criterion(outputs, batch_labels)
    assert not torch.isnan(loss), "Loss is NaN!"
    loss.backward()
    optimizer.step()
    print(f"  Forward pass OK, output shape: {outputs.shape}, loss: {loss.item():.4f}")
    print("  Backward pass + optimizer step OK")

    print("\n[7/7] Testing checkpoint save/load + EarlyStopping...")
    ckpt_path = save_checkpoint(model, optimizer, epoch=1, best_val_metric=0.5,
                                  checkpoint_dir=str(TEST_DIR / "checkpoints"))
    assert Path(ckpt_path).exists(), "Checkpoint file was not created"

    # Build a fresh model and load into it
    model2 = build_model(config)
    checkpoint = load_checkpoint(ckpt_path, model2, device=torch.device("cpu"))
    assert checkpoint["epoch"] == 1, "Checkpoint epoch mismatch"
    print(f"  Checkpoint saved and reloaded successfully")

    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) == False
    assert stopper.step(0.4) == False  # no improvement, counter=1
    assert stopper.step(0.4) == True   # no improvement, counter=2 -> stop
    print("  EarlyStopping logic verified (triggers correctly after patience exceeded)")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✅")
    print("=" * 60)

    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    run_smoke_test()
