"""
End-to-end smoke test for the v2 pipeline on a tiny synthetic dataset, CPU
only, no downloads (random-init resnet18 at 64px). It proves the code runs:
metadata encoding round-trip, training with metadata dropout + EMA, TTA
caches, the image-only ablation, and evaluate_v2's paired comparison against
a v1-style cache (internal + external). It says nothing about accuracy.

Run directly: python tests/v2_smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DX_LABELS, LABEL_TO_IDX, load_metadata, stratified_split  # noqa: E402
from src.v2.metadata import MetadataEncoder  # noqa: E402

TINT = {c: np.array(v) for c, v in zip(DX_LABELS, [(200, 90, 90), (90, 200, 90), (90, 90, 200), (200, 200, 90),
                                                  (60, 40, 30), (180, 140, 120), (200, 60, 160)])}


def make_images(root: Path, prefix: str, n: int, rng) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        dx = DX_LABELS[i % len(DX_LABELS)]
        arr = np.clip(TINT[dx] + rng.normal(0, 25, (48, 64, 3)), 0, 255).astype(np.uint8)
        image_id = f"{prefix}_{i:04d}"
        Image.fromarray(arr).save(root / f"{image_id}.jpg")
        rows.append({
            "lesion_id": f"{prefix}L{i // 2:04d}", "image_id": image_id, "dx": dx,
            "dx_type": "histo", "age": float(rng.integers(20, 85)) if i % 11 else np.nan,
            "sex": ["male", "female", "unknown"][i % 3],
            "localization": ["back", "face", "trunk", "unknown"][i % 4],
        })
    return pd.DataFrame(rows)


def test_metadata_encoder():
    df = pd.DataFrame({"age": [30, np.nan, 70], "sex": ["male", "unknown", "female"],
                       "localization": ["back", "face", None]})
    enc = MetadataEncoder.fit(df)
    assert enc.sexes == ["female", "male"] and enc.sites == ["back", "face"]
    again = MetadataEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert again.columns == enc.columns
    v = enc.encode(None, "unknown", "")
    assert v[enc.columns.index("age_missing")] == 1 and v[enc.columns.index("sex_missing")] == 1
    assert v[enc.columns.index("loc_missing")] == 1 and v[enc.columns.index("age_scaled")] == 0
    full = enc.encode(50, "Male", "face")
    assert full[enc.columns.index("sex_male")] == 1 and full[enc.columns.index("loc_face")] == 1
    assert np.array_equal(enc.encode(50, "male", "face", drop=(True, True, True)), v)
    assert enc.validate("male", "back") == [] and len(enc.validate("robot", "moon")) == 2
    print("[ok] metadata encoder")


def run(cmd, cwd):
    print("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise AssertionError(f"command failed: {' '.join(cmd)}")
    return res.stdout


def test_pipeline():
    tmp = Path(tempfile.mkdtemp(prefix="v2smoke_"))
    try:
        rng = np.random.default_rng(0)
        ham = make_images(tmp / "ham_part1", "HAM", 140, rng)
        ham.to_csv(tmp / "meta.csv", index=False)
        isic = make_images(tmp / "isic", "ISIC", 35, rng)
        isic.drop(columns=["dx_type"]).to_csv(tmp / "isic_gt.tab", sep="\t", index=False)
        (tmp / "ham_part2").mkdir()

        cfg = yaml.safe_load((REPO_ROOT / "configs" / "config_v2.yaml").read_text())
        cfg["data"].update(metadata_csv=str(tmp / "meta.csv"), images_dir_part1=str(tmp / "ham_part1"),
                           images_dir_part2=str(tmp / "ham_part2"))
        cfg["model"].update(backbone="resnet18", pretrained=False)
        cfg["train"].update(image_size=64, batch_size=8, num_epochs=2, num_workers=0, device="cpu",
                            tta_views=2, amp=False)
        cfg["paths"]["output_dir"] = str(tmp / "runs")
        cfg_path = tmp / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        py = sys.executable
        run([py, "-m", "src.train_v2", "--config", str(cfg_path)], REPO_ROOT)
        run([py, "-m", "src.train_v2", "--config", str(cfg_path), "--set", "train.use_metadata=false"], REPO_ROOT)

        fusion = tmp / "runs" / "fusion"
        for f in ["best_model_v2.pt", "summary.json", "history.json", "cache/test_logits.pt", "cache/val_image_ids.json"]:
            assert (fusion / f).exists(), f
        assert (tmp / "runs" / "image_only" / "best_model_v2.pt").exists()
        state = torch.load(fusion / "best_model_v2.pt", weights_only=False)
        assert state["format"] == "cancer-fusion-v2" and state["metadata_encoder"]["version"] == 2

        # Fake "v1" caches in the same image order evaluate_v2 will use, so the
        # paired-comparison path (and its alignment check) is exercised.
        _, _, test_df = stratified_split(load_metadata(str(tmp / "meta.csv")), 0.7, 0.15, 0.15, 42)
        v1_int, v1_ext = tmp / "v1_int", tmp / "v1_ext"
        for cache, prefix, labels in [(v1_int, "test", test_df["dx"]), (v1_ext, "isic2018_test", isic["dx"])]:
            cache.mkdir()
            y = torch.tensor([LABEL_TO_IDX[d] for d in labels])
            torch.save(torch.randn(len(y), 7), cache / f"{prefix}_logits.pt")
            torch.save(y, cache / f"{prefix}_labels.pt")

        out = run([py, "-m", "src.evaluate_v2", "--checkpoint", str(fusion / "best_model_v2.pt"),
                   "--config", str(cfg_path), "--isic-images-dir", str(tmp / "isic"),
                   "--isic-groundtruth", str(tmp / "isic_gt.tab"), "--v1-internal-cache", str(v1_int),
                   "--v1-external-cache", str(v1_ext), "--tta", "2", "--num-workers", "0", "--n-boot", "50"],
                  REPO_ROOT)
        report = json.loads((fusion / "eval" / "report.json").read_text())
        for s in ("internal_test", "external_isic2018"):
            for scenario in ("with_metadata", "metadata_withheld"):
                assert report["sets"][s][scenario]["vs_v1"]["balanced_accuracy"]["ci95"], (s, scenario)
        assert "| external_isic2018 | metadata_withheld | mel_auc |" in out
        print("[ok] train_v2 (fusion + image-only) and evaluate_v2 end to end")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_metadata_encoder()
    test_pipeline()
    print("v2 smoke test passed")
