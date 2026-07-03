"""
Config loader for Cancer Fusion AI.
Fails fast with clear errors if the YAML is malformed or missing required keys,
instead of crashing mysteriously deep inside training.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


REQUIRED_TOP_LEVEL_KEYS = ["data", "model", "train", "paths", "logging"]


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or incomplete."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found at '{config_path}'. "
            f"Did you forget to update the path, or run this from the wrong directory?"
        )

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Config file '{config_path}' is not valid YAML: {e}")

    if config is None:
        raise ConfigError(f"Config file '{config_path}' is empty.")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ConfigError(
            f"Config is missing required top-level section(s): {missing}. "
            f"Check configs/config.yaml against the template."
        )

    # Basic sanity checks that catch typo-level mistakes early
    splits = config["data"]
    total_split = splits.get("train_split", 0) + splits.get("val_split", 0) + splits.get("test_split", 0)
    if not (0.99 <= total_split <= 1.01):
        raise ConfigError(
            f"data.train_split + val_split + test_split must sum to 1.0, got {total_split}"
        )

    if config["model"].get("num_classes", 0) <= 0:
        raise ConfigError("model.num_classes must be a positive integer")

    if config["train"].get("batch_size", 0) <= 0:
        raise ConfigError("train.batch_size must be a positive integer")

    return config
