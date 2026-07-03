"""
Edge case tests — deliberately break things to verify the code fails
gracefully with clear error messages, instead of crashing with a
confusing stack trace mid-training.
"""
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, ConfigError
from src.dataset import load_metadata, HAM10000Dataset, DX_LABELS

TEST_DIR = Path("/tmp/ham10000_edge_test")


def setup():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)


def test_missing_config_file():
    print("[test] Missing config file...")
    try:
        load_config("/tmp/this_config_does_not_exist.yaml")
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "not found" in str(e)
        print(f"  PASS: raised clear ConfigError -> {e}")


def test_malformed_yaml():
    print("[test] Malformed YAML...")
    bad_yaml = TEST_DIR / "bad.yaml"
    bad_yaml.write_text("data: [this is: not: valid: yaml:")
    try:
        load_config(str(bad_yaml))
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        print(f"  PASS: raised clear ConfigError -> {e}")


def test_config_missing_required_keys():
    print("[test] Config missing required sections...")
    incomplete_yaml = TEST_DIR / "incomplete.yaml"
    incomplete_yaml.write_text("data:\n  train_split: 0.7\n  val_split: 0.15\n  test_split: 0.15\n")
    try:
        load_config(str(incomplete_yaml))
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "missing required" in str(e)
        print(f"  PASS: raised clear ConfigError -> {e}")


def test_splits_dont_sum_to_one():
    print("[test] Splits summing to != 1.0...")
    bad_split_yaml = TEST_DIR / "bad_split.yaml"
    bad_split_yaml.write_text("""
data:
  train_split: 0.7
  val_split: 0.5
  test_split: 0.15
model:
  num_classes: 7
train:
  batch_size: 32
paths: {}
logging: {}
""")
    try:
        load_config(str(bad_split_yaml))
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "sum to 1.0" in str(e)
        print(f"  PASS: raised clear ConfigError -> {e}")


def test_metadata_missing_columns():
    print("[test] Metadata CSV missing required columns...")
    csv_path = TEST_DIR / "bad_metadata.csv"
    pd.DataFrame({"image_id": ["a", "b"], "wrong_col": [1, 2]}).to_csv(csv_path, index=False)
    try:
        load_metadata(str(csv_path))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "missing required columns" in str(e)
        print(f"  PASS: raised clear ValueError -> {e}")


def test_metadata_unknown_label():
    print("[test] Metadata CSV with an unknown 'dx' label (typo)...")
    csv_path = TEST_DIR / "typo_metadata.csv"
    pd.DataFrame({"image_id": ["a", "b"], "dx": ["mel", "melanomaa"]}).to_csv(csv_path, index=False)
    try:
        load_metadata(str(csv_path))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unexpected" in str(e)
        print(f"  PASS: raised clear ValueError -> {e}")


def test_dataset_with_missing_images():
    print("[test] Dataset where some images referenced in CSV don't exist on disk...")
    img_dir = TEST_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    # Only create image for row 0, not row 1 (simulates a missing/corrupted download)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8)).save(img_dir / "img0.jpg")

    df = pd.DataFrame({"image_id": ["img0", "img1_missing"], "dx": ["mel", "nv"]})
    dataset = HAM10000Dataset(df, [str(img_dir)], ".jpg", transform=None)

    # Should have silently skipped the missing one, not crashed
    assert len(dataset) == 1, f"Expected 1 valid sample (missing one skipped), got {len(dataset)}"
    print(f"  PASS: dataset skipped missing image gracefully, {len(dataset)} valid sample(s) remain")


def test_dataset_with_corrupt_image():
    print("[test] Dataset where an image file is corrupt (unreadable)...")
    img_dir = TEST_DIR / "corrupt_images"
    img_dir.mkdir(exist_ok=True)

    # Write garbage bytes instead of a real image
    (img_dir / "corrupt.jpg").write_bytes(b"this is not a real jpeg file")

    df = pd.DataFrame({"image_id": ["corrupt"], "dx": ["nv"]})
    dataset = HAM10000Dataset(df, [str(img_dir)], ".jpg", transform=None)
    # File exists (so it's not filtered at init time) but is unreadable -> should fall back, not crash
    image, label = dataset[0]
    assert image is not None, "Should return a fallback image instead of crashing"
    print(f"  PASS: corrupt image handled with fallback, no crash")


def test_dataset_all_images_missing():
    print("[test] Dataset where ALL referenced images are missing (should raise clearly)...")
    empty_dir = TEST_DIR / "empty"
    empty_dir.mkdir(exist_ok=True)
    df = pd.DataFrame({"image_id": ["nope1", "nope2"], "dx": ["mel", "nv"]})
    try:
        HAM10000Dataset(df, [str(empty_dir)], ".jpg", transform=None)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No valid images found" in str(e)
        print(f"  PASS: raised clear RuntimeError -> {e}")


def run_all():
    print("=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)
    setup()
    test_missing_config_file()
    test_malformed_yaml()
    test_config_missing_required_keys()
    test_splits_dont_sum_to_one()
    test_metadata_missing_columns()
    test_metadata_unknown_label()
    test_dataset_with_missing_images()
    test_dataset_with_corrupt_image()
    test_dataset_all_images_missing()
    shutil.rmtree(TEST_DIR)
    print("\n" + "=" * 60)
    print("ALL EDGE CASE TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
