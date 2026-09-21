"""
Tests for the OOD guard + conformal prediction set + malignant-sensitivity
rule wired into POST /predict (see app.py). Boots the real app (real
checkpoint, real artifacts) via FastAPI's TestClient — these are slower than
the other tests/ files (they load a real ResNet50) but they're the only way
to test the actual serving path, not a reimplementation of it.

Run directly: python tests/predict_guards_test.py
"""
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)  # app.py uses paths relative to repo root

REPO_ROOT = Path(__file__).parent.parent
HAM_IMAGES_DIR = "/Users/avnishsingh1505/Downloads/archive/HAM10000_images_part_1"


def _make_noise_image_bytes(seed=0, size=(224, 224)):
    rng = np.random.default_rng(seed)
    arr = (rng.random((*size, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _real_dermoscopic_image_path():
    df = pd.read_csv(REPO_ROOT / "data" / "HAM10000_metadata.csv")
    for image_id in df["image_id"]:
        p = Path(HAM_IMAGES_DIR) / f"{image_id}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"No HAM10000 images found under {HAM_IMAGES_DIR} — set HAM_IMAGES_DIR.")


def _client():
    import app as appmod
    from fastapi.testclient import TestClient
    return TestClient(appmod.app), appmod


def test_accepted_response_schema():
    print("[test] Accepted response has the documented schema...")
    client, _ = _client()
    img_path = _real_dermoscopic_image_path()
    with open(img_path, "rb") as f:
        r = client.post(
            "/predict?explain=false",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"age": 50, "sex": "male", "localization": "back"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    for key in ("prediction", "prediction_full_name", "confidence", "all_probabilities",
                "prediction_set", "malignant", "ood", "timings_ms"):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["prediction_set"], list) and len(body["prediction_set"]) >= 1
    assert set(body["all_probabilities"].keys()) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert isinstance(body["malignant"]["flagged"], bool)
    assert body["malignant"]["operating_point"] == "95% sensitivity on validation"
    assert {"probability", "flagged", "operating_point"} <= set(body["malignant"].keys())
    assert {"distance", "threshold"} <= set(body["ood"].keys())
    assert "message" not in body, "accepted responses should not carry a rejection message"
    print("  PASS")


def test_rejected_response_schema():
    print("[test] Rejected response has the documented schema and omits prediction fields...")
    client, _ = _client()
    r = client.post(
        "/predict?explain=false",
        files={"file": ("noise.jpg", _make_noise_image_bytes(seed=1), "image/jpeg")},
        data={"age": 50, "sex": "male", "localization": "back"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "rejected"
    assert "message" in body and len(body["message"]) > 0
    assert {"distance", "threshold"} <= set(body["ood"].keys())
    for forbidden in ("prediction", "prediction_full_name", "confidence", "all_probabilities",
                      "prediction_set", "malignant"):
        assert forbidden not in body, f"rejected response leaked '{forbidden}'"
    print("  PASS")


def test_noise_image_is_rejected():
    print("[test] A random-noise image is rejected by the OOD guard...")
    client, _ = _client()
    rejected_count = 0
    for seed in range(5):
        r = client.post(
            "/predict?explain=false",
            files={"file": (f"noise{seed}.jpg", _make_noise_image_bytes(seed=seed), "image/jpeg")},
            data={"age": 50, "sex": "male", "localization": "back"},
        )
        assert r.status_code == 200, r.text
        if r.json()["status"] == "rejected":
            rejected_count += 1
    assert rejected_count == 5, f"expected all 5 noise images rejected, got {rejected_count}/5"
    print("  PASS (5/5 noise images rejected)")


def test_known_dermoscopic_image_is_accepted():
    print("[test] A known real dermoscopic test image is accepted...")
    client, _ = _client()
    img_path = _real_dermoscopic_image_path()
    with open(img_path, "rb") as f:
        r = client.post(
            "/predict?explain=false",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"age": 50, "sex": "male", "localization": "back"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok", f"expected a real HAM10000 image to be accepted, got: {body}"
    assert body["ood"]["distance"] < body["ood"]["threshold"]
    print(f"  PASS ({img_path.name} accepted, distance={body['ood']['distance']:.1f} < "
          f"threshold={body['ood']['threshold']:.1f})")


def test_startup_fails_on_checkpoint_hash_mismatch():
    print("[test] Server refuses to start if the checkpoint doesn't match the artifacts file...")
    real_artifacts_path = REPO_ROOT / "models" / "inference_artifacts.json"
    artifacts = json.loads(real_artifacts_path.read_text())
    artifacts["checkpoint_sha256"] = "0" * 64  # deliberately wrong

    tmp_dir = REPO_ROOT / "tests" / "_tmp_bad_artifacts"
    tmp_dir.mkdir(exist_ok=True)
    bad_artifacts_path = tmp_dir / "inference_artifacts.json"
    bad_artifacts_path.write_text(json.dumps(artifacts))
    # npz referenced by the artifacts file needs to exist alongside it too,
    # even though we expect the hash check to fail before it's read.
    shutil.copy(real_artifacts_path.parent / artifacts["ood_guard"]["means_precision_file"],
                tmp_dir / artifacts["ood_guard"]["means_precision_file"])

    env = {**os.environ, "INFERENCE_ARTIFACTS_PATH": str(bad_artifacts_path)}
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    shutil.rmtree(tmp_dir)

    assert result.returncode != 0, "app.py should have refused to start on a checkpoint/artifact hash mismatch"
    assert "does not match the checkpoint" in result.stderr, result.stderr[-2000:]
    print("  PASS (non-zero exit, clear error message)")


def test_predictions_match_pre_change_output():
    """
    Reconstructs the OLD (pre-guard) computation directly — single
    model(image, meta) call at the OLD hardcoded T=2.1235 — and compares
    against the NEW app.py's decomposed forward_features/classifier_head
    path at the SAME temperature. These must be bit-identical: the guard
    logic sits *around* the classifier, it must not change what the
    classifier itself computes. (The temperature source also changed,
    separately and by design — see the conversation's equivalence proof;
    that is intentionally NOT what this test checks.)
    """
    print("[test] Predictions match the pre-change computation (decomposition equivalence)...")
    import torch
    import app as appmod

    OLD_TEMPERATURE = 2.1235
    img_path = _real_dermoscopic_image_path()
    pil_image = Image.open(img_path).convert("RGB")
    image_tensor = appmod.transform(pil_image).unsqueeze(0).to(appmod.device)
    metadata_tensor = appmod.encode_metadata(50, "male", "back")

    with torch.no_grad():
        output_old = appmod.model(image_tensor, metadata_tensor)
        probs_old = (output_old / OLD_TEMPERATURE).softmax(dim=1)[0]

        image_features = appmod.model.forward_features(image_tensor)
        meta_features = appmod.model.metadata_processor(metadata_tensor)
        fused = torch.cat((image_features, meta_features), dim=1)
        output_new = appmod.model.classifier_head(fused)
        probs_new = (output_new / OLD_TEMPERATURE).softmax(dim=1)[0]

    assert torch.equal(output_old, output_new), "logits differ between old and new code paths"
    assert torch.equal(probs_old, probs_new), "probabilities differ between old and new code paths"
    assert probs_old.argmax().item() == probs_new.argmax().item()
    print("  PASS (bit-identical logits and probabilities)")


def run_all():
    print("=" * 60)
    print("PREDICT GUARD TESTS (OOD guard, conformal set, malignant rule)")
    print("=" * 60)
    test_accepted_response_schema()
    test_rejected_response_schema()
    test_noise_image_is_rejected()
    test_known_dermoscopic_image_is_accepted()
    test_startup_fails_on_checkpoint_hash_mismatch()
    test_predictions_match_pre_change_output()
    print("\n" + "=" * 60)
    print("ALL PREDICT GUARD TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
