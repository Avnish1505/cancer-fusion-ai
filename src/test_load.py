import torch
from src.config import load_config
from src.dataset import HAM10000Dataset, load_metadata
from src.model import build_model
from src.utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = load_config("configs/config.yaml")
use_metadata = config["train"].get("use_metadata", False)

if use_metadata:
    # To get the correct metadata_dim, we must initialize a dataset object
    # just like in training. This ensures the model architecture is identical.
    df = load_metadata(config["data"]["metadata_csv"])
    image_dirs = [config["data"]["images_dir_part1"], config["data"]["images_dir_part2"]]
    train_ds = HAM10000Dataset(df, image_dirs, use_metadata=True)
    config["model"]["metadata_dim"] = len(train_ds.metadata_columns)
    print(f"Metadata is enabled. Dimension set to: {config['model']['metadata_dim']}")

model = build_model(config).to(device)

checkpoint_path = f"{config['paths']['checkpoint_dir']}/best_model.pt"
load_checkpoint(checkpoint_path, model, device=device)
model.eval()

print("Model loaded successfully!")
print(model.classifier_head)