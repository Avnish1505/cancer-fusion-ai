"""
Phase 1 training entry point: image-only baseline on HAM10000.

Usage (on Kaggle/Colab, after cloning this repo and installing requirements):
    python -m src.train --config configs/config.yaml
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.config import load_config, ConfigError
from src.dataset import load_metadata, stratified_split, HAM10000Dataset, compute_class_weights
from src.transforms import get_train_transforms, get_eval_transforms
from src.model import build_model
from src.utils import set_seed, get_device, save_checkpoint, EarlyStopping


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_train: bool, use_metadata: bool):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(is_train)
    for batch in dataloader:
        if use_metadata:
            images, metas, labels = batch
            images, metas, labels = images.to(device), metas.to(device), labels.to(device)
            
            outputs = model(images, metas)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

        if is_train:
            optimizer.zero_grad()
        
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["data"]["random_seed"])
    device = get_device(config["train"]["device"])
    print(f"[train] Using device: {device}")

    # ---- Data ----
    df = load_metadata(config["data"]["metadata_csv"])
    train_df, val_df, _ = stratified_split(
        df,
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
        config["data"]["random_seed"],
    )

    image_dirs = [config["data"]["images_dir_part1"], config["data"]["images_dir_part2"]]
    image_size = config["train"]["image_size"]
    use_metadata = config["train"].get("use_metadata", False)
    
    train_dataset = HAM10000Dataset(
        train_df, image_dirs, config["data"]["image_extension"],
        transform=get_train_transforms(image_size),
        use_metadata=use_metadata,
    )
    
    # For val/test, reuse the scaler and column order from the training set
    age_scaler = train_dataset.age_scaler if use_metadata else None
    metadata_columns = train_dataset.metadata_columns if use_metadata else None
    if use_metadata:
        print(f"[train] Using metadata. Dimension: {len(metadata_columns)}")

    val_dataset = HAM10000Dataset(
        val_df, image_dirs, config["data"]["image_extension"],
        transform=get_eval_transforms(image_size),
        use_metadata=use_metadata,
        age_scaler=age_scaler,
        metadata_columns=metadata_columns,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True,
        num_workers=config["train"]["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"]["num_workers"], pin_memory=(device.type == "cuda"),
    )

    # ---- Model, loss, optimizer ----
    model = build_model(config).to(device)

    if config["train"]["use_class_weights"]:
        class_weights = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[train] Using class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    early_stopper = EarlyStopping(patience=config["train"]["early_stopping_patience"], mode="max")

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0

    # ---- Training loop ----
    for epoch in range(1, config["train"]["num_epochs"] + 1):
        start_time = time.time()

        train_loss, train_acc, train_f1 = run_one_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True, use_metadata=use_metadata
        )
        val_loss, val_acc, val_f1 = run_one_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False, use_metadata=use_metadata
        )

        elapsed = time.time() - start_time
        print(
            f"[Epoch {epoch}/{config['train']['num_epochs']}] "
            f"({elapsed:.1f}s) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = save_checkpoint(
                model, optimizer, epoch, best_val_f1, checkpoint_dir, filename="best_model.pt"
            )
            print(f"[train] New best model (val_f1={val_f1:.4f}) saved to {ckpt_path}")

        if early_stopper.step(val_f1):
            print(f"[train] Early stopping triggered at epoch {epoch} "
                  f"(no improvement in val_f1 for {early_stopper.patience} epochs).")
            break

    print(f"[train] Training complete. Best val_f1: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 1 image-only baseline on HAM10000")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                         help="Path to config YAML file")
    args = parser.parse_args()

    try:
        main(args.config)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"[FILE ERROR] {e}")
        raise SystemExit(1)
