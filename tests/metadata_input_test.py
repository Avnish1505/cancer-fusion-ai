"""
Input-contract tests for POST /predict (boots the real app with the real
checkpoint, like tests/predict_guards_test.py).

Covers the change that removed the silent 50 / "male" / "back" metadata
defaults: metadata is now required, unrecognised categories are rejected
instead of being encoded as an all-zero row the model never saw, and an
unreadable upload is a 400 instead of a 500.

Run directly: python tests/metadata_input_test.py
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)


def _noise_jpeg(seed=0):
    arr = (np.random.default_rng(seed).random((224, 224, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _client():
    from fastapi.testclient import TestClient
    import app as app_module
    return TestClient(app_module.app), app_module


def test_missing_metadata_is_rejected(client):
    r = client.post("/predict", files={"file": ("x.jpg", _noise_jpeg(), "image/jpeg")})
    assert r.status_code == 422, r.text
    print("[ok] missing metadata -> 422")


def test_unrecognised_categories_are_rejected(client):
    r = client.post("/predict", files={"file": ("x.jpg", _noise_jpeg(), "image/jpeg")},
                    data={"age": "40", "sex": "robot", "localization": "moon"})
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["status"] == "invalid_input" and len(body["errors"]) == 2
    assert "back" in body["allowed"]["localization"] and "male" in body["allowed"]["sex"]
    print("[ok] unrecognised sex/localization -> 422 with allowed values")


def test_bad_age_is_rejected(client):
    for bad in ["-3", "200", "forty"]:
        r = client.post("/predict", files={"file": ("x.jpg", _noise_jpeg(), "image/jpeg")},
                        data={"age": bad, "sex": "male", "localization": "back"})
        assert r.status_code == 422, (bad, r.text)
    print("[ok] out-of-range / non-numeric age -> 422")


def test_unreadable_file_is_400(client):
    r = client.post("/predict", files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
                    data={"age": "unknown", "sex": "unknown", "localization": "unknown"})
    assert r.status_code == 400, r.text
    print("[ok] unreadable upload -> 400")


def test_explicit_unknowns_and_case_are_accepted(client):
    # Noise is rejected by the OOD guard, but only AFTER input validation
    # passed, so a 200 'rejected' proves the metadata was accepted.
    r = client.post("/predict", files={"file": ("x.jpg", _noise_jpeg(3), "image/jpeg")},
                    data={"age": "Unknown", "sex": " Female ", "localization": "Lower Extremity"})
    assert r.status_code == 200 and r.json()["status"] == "rejected", r.text
    print("[ok] explicit unknown age + mixed-case categories accepted")


def test_unknown_age_encodes_as_training_median(app_module):
    t = app_module.encode_metadata(None, "unknown", "unknown")[0]
    cols = app_module.metadata_columns
    expected = app_module.age_scaler.transform([[app_module.age_median]])[0][0]
    assert abs(float(t[cols.index("age_scaled")]) - expected) < 1e-6
    assert t[cols.index("sex_unknown")] == 1 and t[cols.index("loc_unknown")] == 1
    print("[ok] unknown age -> training-split median, same as training imputation")


if __name__ == "__main__":
    client, module = _client()
    test_missing_metadata_is_rejected(client)
    test_unrecognised_categories_are_rejected(client)
    test_bad_age_is_rejected(client)
    test_unreadable_file_is_400(client)
    test_explicit_unknowns_and_case_are_accepted(client)
    test_unknown_age_encodes_as_training_median(module)
    print("metadata input tests passed")
