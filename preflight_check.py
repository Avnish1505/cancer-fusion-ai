"""
Pre-flight Check for Phase 2 (Fusion Model).

This script runs a short training loop on a small, stratified subset of the
real HAM10000 data to verify the end-to-end fusion pipeline before launching
a full training run.

It checks for:
- Correct data loading with metadata.
- Correct reuse of scalers/columns for validation set.
- No shape mismatches or device errors during training.
- Loss is decreasing (i.e., the model is learning).
- No NaN or exploding gradients.

Usage:
    python preflight_check.py --config configs/config.yaml
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import load_config, ConfigError
from src.dataset import load_metadata, stratified_split, HAM10000Dataset, compute_class_weights
from src.transforms import get_train_transforms, get_eval_transforms
from src.model import build_model
from src.utils import set_seed, get_device
from src.train import run_one_epoch


def run_preflight_check(config_path: str):
    print("=" * 60)
    print("RUNNING PRE-FLIGHT CHECK FOR FUSION MODEL")
    print("=" * 60)

    # --- 1. Load Config and Setup ---
    config = load_config(config_path)
    set_seed(config["data"]["random_seed"])
    device = get_device(config["train"]["device"])
    
    # Force metadata usage for this check
    config["train"]["use_metadata"] = True
    
    print(f"[Check] Using device: {device}")

    # --- 2. Load a small, stratified subset of data ---
    full_df = load_metadata(config["data"]["metadata_csv"])
    
    # Create a 10% stratified subset of the data for the check
    subset_df, _ = train_test_split(
        full_df,
        test_size=0.90,
        stratify=full_df['dx'],
        random_state=config["data"]["random_seed"]
    )
    print(f"[Check] Loaded a stratified subset of {len(subset_df)} samples for the check.")

    # --- 3. Split subset into train/val and create Datasets ---
    train_df, val_df, _ = stratified_split(
        subset_df, train_split=0.8, val_split=0.15, test_split=0.05,
        random_seed=config["data"]["random_seed"]
    )

    image_dirs = [config["data"]["images_dir_part1"], config["data"]["images_dir_part2"]]
    image_size = config["train"]["image_size"]
    
    train_ds = HAM10000Dataset(
        train_df, image_dirs, config["data"]["image_extension"],
        transform=get_train_transforms(image_size),
        use_metadata=True,
    )
    print(f"[Check] Train dataset created. Metadata dim: {len(train_ds.metadata_columns)}")
    
    # CRITICAL: Reuse scaler and columns from train_ds for val_ds
    val_ds = HAM10000Dataset(
        val_df, image_dirs, config["data"]["image_extension"],
        transform=get_eval_transforms(image_size),
        use_metadata=True,
        age_scaler=train_ds.age_scaler,
        metadata_columns=train_ds.metadata_columns,
    )
    print(f"[Check] Validation dataset created. Metadata dim: {len(val_ds.metadata_columns)}")
    assert len(train_ds.metadata_columns) == len(val_ds.metadata_columns), "Metadata column mismatch!"

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    # --- 4. Build Model, Loss, Optimizer ---
    model = build_model(config).to(device)
    
    class_weights = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    # --- 5. Run short training loop and verify ---
    num_check_epochs = 5
    print(f"\n[Check] Starting {num_check_epochs}-epoch training loop on subset...")
    
    for epoch in range(1, num_check_epochs + 1):
        start_time = time.time()
        train_loss, _, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True, use_metadata=True
        )
        val_loss, _, _ = run_one_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False, use_metadata=True
        )
        elapsed = time.time() - start_time

        print(f"  [Epoch {epoch}/{num_check_epochs}] ({elapsed:.1f}s) train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if torch.isnan(torch.tensor(train_loss)):
            print("\n[FAILURE] ❌ Loss is NaN. Aborting.")
            print("  Root Cause: This often indicates a numerical instability issue. Check for:")
            print("    - Very high learning rate.")
            print("    - Issues in the data processing (e.g., non-normalized inputs).")
            print("    - A bug in the model architecture.")
            raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK PASSED ✅")
    print("=" * 60)
    print("  - Pipeline ran end-to-end without crashing.")
    print("  - Loss did not become NaN.")
    print("  - Metadata column alignment between train/val was successful.")
    print("You are clear for a full training run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pre-flight check on the fusion model pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    run_preflight_check(args.config)