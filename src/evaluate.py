"""
Evaluation script for the trained Phase 1 model.
Produces: classification report, confusion matrix (saved as PNG),
and highlights per-class recall — critical in medical AI, since a
missed melanoma (false negative) is far costlier than a false alarm.

Usage:
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
"""
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config, ConfigError
from src.dataset import load_metadata, stratified_split, HAM10000Dataset, DX_LABELS, DX_FULL_NAMES
from src.transforms import get_eval_transforms
from src.model import build_model
from src.utils import get_device, load_checkpoint


def evaluate(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device(config["train"]["device"])

    use_metadata = config["train"].get("use_metadata", False)

    df = load_metadata(config["data"]["metadata_csv"])
    train_df, val_df, test_df = stratified_split(
        df,
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
        config["data"]["random_seed"],
    )
    image_dirs = [config["data"]["images_dir_part1"], config["data"]["images_dir_part2"]]

    # We need to initialize a train_dataset to get the fitted age_scaler and metadata_columns
    # to ensure the test set is processed identically.
    train_dataset = HAM10000Dataset(
        train_df, image_dirs, use_metadata=use_metadata,
        transform=get_eval_transforms(config["train"]["image_size"]) # transform doesn't matter here
    )
    age_scaler = train_dataset.age_scaler if use_metadata else None
    age_median = train_dataset.age_median if use_metadata else None
    metadata_columns = train_dataset.metadata_columns if use_metadata else None
    if use_metadata:
        # Set metadata_dim in config so build_model creates the correct architecture
        config["model"]["metadata_dim"] = len(metadata_columns)

    test_dataset = HAM10000Dataset(
        test_df, image_dirs, config["data"]["image_extension"],
        transform=get_eval_transforms(config["train"]["image_size"]),
        use_metadata=use_metadata,
        age_scaler=age_scaler,
        age_median=age_median,
        metadata_columns=metadata_columns,
    )

    if use_metadata:
        print(f"[evaluate] Using metadata. Dimension: {len(metadata_columns)}")

    test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # --- Debugging: Inspect the first batch from the test_loader ---
    print("\n[evaluate] Inspecting first batch from test_loader...")
    try:
        batch = next(iter(test_loader))
        print(f"  Batch contains {len(batch)} items.")
        for i, item in enumerate(batch):
            print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
    except StopIteration:
        print("  [ERROR] test_loader is empty!")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if use_metadata:
                images, metas, labels = batch
                images, metas = images.to(device), metas.to(device)
            else:
                images, labels = batch
                images = images.to(device)

            outputs = model(images, metas) if use_metadata else model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        all_labels, all_preds, target_names=DX_LABELS, digits=3, zero_division=0
    )
    print(report)

    # Flag the medically critical class specifically
    mel_idx = DX_LABELS.index("mel")
    mel_recall = sum(
        1 for p, l in zip(all_preds, all_labels) if l == mel_idx and p == mel_idx
    ) / max(sum(1 for l in all_labels if l == mel_idx), 1)
    print(f"\n[IMPORTANT] Melanoma (mel) recall: {mel_recall:.3f}")
    print("  -> This is the most clinically critical number: it's the fraction of actual")
    print("     melanoma cases the model correctly caught. Low recall here means missed")
    print("     cancers, which matters far more than overall accuracy.")

    # Confusion matrix
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DX_LABELS, yticklabels=DX_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("HAM10000 Phase 1 - Confusion Matrix")
    plt.tight_layout()

    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\n[evaluate] Confusion matrix saved to {cm_path}")

    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\n\nMelanoma recall: {mel_recall:.3f}\n")
    print(f"[evaluate] Classification report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Phase 1 model on HAM10000 test set")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    try:
        evaluate(args.config, args.checkpoint)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"[FILE ERROR] {e}")
        raise SystemExit(1)
