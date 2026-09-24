import base64
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
torch.set_num_threads(1)  # single-worker container on a shared vCPU — don't let torch oversubscribe it

import pandas as pd

matplotlib.use('Agg')
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from sklearn.preprocessing import StandardScaler
from src.config import load_config
from src.dataset import DX_FULL_NAMES, DX_LABELS, load_metadata, stratified_split
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint

app = FastAPI(title="Cancer Fusion AI API")

# Captured as early as possible so /readyz reports true process uptime,
# including the cold-start work below (config/checkpoint/scaler loading).
PROCESS_START = time.time()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cancer-fusion-ai.vercel.app",  # Vercel URL ke liye isko update karna padega
        "http://localhost:5173",  # local Vite dev server
        "http://127.0.0.1:5173",  # local Vite dev server (127.0.0.1 form)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Load model once at startup ----
config = load_config("configs/config.yaml")

CHECKPOINT_PATH = "models/best_model.pt"
# Overridable so tests can point at a throwaway artifacts file without
# touching the real one — see tests/test_predict_guards.py.
ARTIFACTS_PATH = os.environ.get("INFERENCE_ARTIFACTS_PATH", "models/inference_artifacts.json")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Frozen inference artifacts: temperature, Mondrian-LAC conformal ----
# ---- quantiles, malignant-sensitivity thresholds, OOD guard params. ----
# ---- See src/export_artifacts.py — every number here was computed by ----
# ---- that script from the analysis caches, never hand-typed. Refuses ----
# ---- to start if the checkpoint on disk doesn't match what they were ----
# ---- computed from, so a swapped checkpoint can't silently serve ----
# ---- stale thresholds. ----
if not Path(ARTIFACTS_PATH).exists():
    raise RuntimeError(
        f"Inference artifacts not found at '{ARTIFACTS_PATH}'. Run "
        f"`python -m src.export_artifacts` before starting the server."
    )
with open(ARTIFACTS_PATH) as f:
    artifacts = json.load(f)

_checkpoint_sha256 = sha256_of(CHECKPOINT_PATH)
if _checkpoint_sha256 != artifacts["checkpoint_sha256"]:
    raise RuntimeError(
        f"'{CHECKPOINT_PATH}' (sha256={_checkpoint_sha256}) does not match the checkpoint "
        f"'{ARTIFACTS_PATH}' was computed from (sha256={artifacts['checkpoint_sha256']}). "
        f"Re-run `python -m src.export_artifacts` before starting the server."
    )

if artifacts["class_order"] != DX_LABELS:
    raise RuntimeError(
        f"'{ARTIFACTS_PATH}' class_order {artifacts['class_order']} does not match "
        f"src.dataset.DX_LABELS {DX_LABELS}."
    )

TEMPERATURE = artifacts["temperature"]
CONFORMAL_QHAT = np.array([artifacts["conformal"]["per_class_qhat"][c] for c in DX_LABELS])
MALIGNANT_CLASSES = artifacts["malignant_rule"]["classes"]
MALIGNANT_IDX = [DX_LABELS.index(c) for c in MALIGNANT_CLASSES]
MALIGNANT_THRESHOLD = artifacts["malignant_rule"]["thresholds"]["sensitivity_95"]["threshold"]
MALIGNANT_OPERATING_POINT = "95% sensitivity on validation"
OOD_THRESHOLD = artifacts["ood_guard"]["threshold"]

_maha_npz_path = Path(ARTIFACTS_PATH).parent / artifacts["ood_guard"]["means_precision_file"]
_maha_params = np.load(_maha_npz_path)
MAHALANOBIS_MEANS = _maha_params["means"]
MAHALANOBIS_PRECISION = _maha_params["precision"]

REJECTION_MESSAGE = (
    "This doesn't look like a dermoscopic image of a skin lesion. Please upload a "
    "dermatoscope image. If you are worried about a lesion, see a clinician."
)


# ---- Recreate metadata encoding exactly as training did ----
# NOTE: Yeh path aapke local setup ke hisaab se adjust karna pad sakta hai
# CRITICAL: the age scaler / median / one-hot columns must be fit on the SAME
# train split training used (src/dataset.py's HAM10000Dataset does this and
# reuses it for val/test to prevent leakage) — refitting on the full CSV here
# would silently drift from what the checkpoint was actually trained on.
full_metadata_csv = load_metadata("data/HAM10000_metadata.csv")
train_metadata_df, _, _ = stratified_split(
    full_metadata_csv,
    config["data"]["train_split"],
    config["data"]["val_split"],
    config["data"]["test_split"],
    config["data"]["random_seed"],
)

age_median = train_metadata_df["age"].median()
train_metadata_df = train_metadata_df.copy()
train_metadata_df["age"] = train_metadata_df["age"].fillna(age_median)
age_scaler = StandardScaler()
age_scaler.fit(train_metadata_df[["age"]])

sex_dummies = pd.get_dummies(train_metadata_df["sex"].fillna("unknown"), prefix="sex", dtype=float)
loc_dummies = pd.get_dummies(train_metadata_df["localization"].fillna("unknown"), prefix="loc", dtype=float)
metadata_columns = ["age_scaled"] + list(sex_dummies.columns) + list(loc_dummies.columns)

# --- CRITICAL: Dynamically set metadata_dim before building the model ---
config["model"]["metadata_dim"] = len(metadata_columns)

device = get_device(config["train"]["device"])
model = build_model(config)
load_checkpoint(CHECKPOINT_PATH, model, device=device)
model.eval()
model.to(device)

def encode_metadata(age: float, sex: str, localization: str):
    age_val = age if age is not None else age_median
    age_scaled = age_scaler.transform([[age_val]])[0][0]

    row = {col: 0.0 for col in metadata_columns}
    row["age_scaled"] = age_scaled
    sex_col = f"sex_{sex}"
    if sex_col in row:
        row[sex_col] = 1.0
    loc_col = f"loc_{localization}"
    if loc_col in row:
        row[loc_col] = 1.0

    vector = [row[col] for col in metadata_columns]
    return torch.tensor([vector], dtype=torch.float32).to(device)

transform = get_eval_transforms(config["train"]["image_size"])


def mahalanobis_distance(image_features: torch.Tensor) -> float:
    """
    Min-over-class Mahalanobis distance on the image encoder's pooled 2048-d
    feature (image branch only — no metadata). Means/precision are fit on
    training-split features with Ledoit-Wolf shrinkage covariance; see
    src/ood.py:fit_mahalanobis (canonical) and src/export_artifacts.py
    (freezes the fitted values this loads). Reimplemented here (not imported
    from src.ood) to avoid pulling matplotlib.pyplot/sklearn.covariance into
    the serving process's cold start for a ~6-line computation.
    """
    feats = image_features.detach().cpu().numpy().astype(np.float64)[0]
    diffs = feats[None, :] - MAHALANOBIS_MEANS
    dists = np.einsum("kj,jl,kl->k", diffs, MAHALANOBIS_PRECISION, diffs)
    return float(dists.min())


def conformal_prediction_set(probs: torch.Tensor) -> list[str]:
    """Mondrian-LAC, alpha=0.10: include class c if (1 - p(c)) <= qhat[c],
    sorted by probability descending. qhat is per-CANDIDATE-class, calibrated
    on internal validation only — see src/conformal.py."""
    probs_np = probs.detach().cpu().numpy()
    lac_scores = 1.0 - probs_np
    included = [i for i in range(len(DX_LABELS)) if lac_scores[i] <= CONFORMAL_QHAT[i]]
    included.sort(key=lambda i: -probs_np[i])
    return [DX_LABELS[i] for i in included]


class FusionGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_from_features(self, image_features, metadata_tensor, class_idx=None):
        """Takes an already-computed image_features (from model.forward_features,
        called once by the caller — this is what makes the OOD guard and the
        prediction share a single forward pass through the encoder) and runs
        the rest of the model (metadata fusion + classifier) plus the
        backward pass for Grad-CAM. Mathematically identical to the previous
        self.model(image_tensor, metadata_tensor) call — same ops, same
        order — just split so the caller can inspect image_features first."""
        if self.model.use_metadata:
            meta_features = self.model.metadata_processor(metadata_tensor)
            fused_features = torch.cat((image_features, meta_features), dim=1)
            output = self.model.classifier_head(fused_features)
        else:
            output = self.model.classifier_head(image_features)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        # class_idx above is argmax on the raw (untouched) logits — dividing
        # by TEMPERATURE only reshapes the displayed distribution, it can't
        # flip the top-1 prediction.
        return heatmap, class_idx, (output / TEMPERATURE).softmax(dim=1)[0]


gradcam = FusionGradCAM(model, model.encoder.layer4[-1])

# FusionGradCAM stashes forward/backward results as mutable instance state
# (self.activations / self.gradients on the shared `gradcam` object, and the
# shared `model`'s own hooks). Two requests racing through a forward+backward
# concurrently could read each other's gradients. Serialize access to be safe
# regardless of worker/event-loop scheduling assumptions.
model_lock = threading.Lock()


def overlay_heatmap(heatmap, original_img: Image.Image, alpha=0.4):
    img = np.array(original_img.resize((224, 224)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)


def image_to_base64(np_img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    # Zero inference, zero imports/loads — just reports on state that was
    # already established at module import time (see PROCESS_START above).
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "uptime_s": time.time() - PROCESS_START,
        "pid": os.getpid(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(50),
    sex: str = Form("male"),
    localization: str = Form("back"),
    explain: bool = True,
):
    timings_ms = {}

    t0 = time.perf_counter()
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t1 = time.perf_counter()
    timings_ms["image_decode"] = (t1 - t0) * 1000

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    use_metadata = config["train"].get("use_metadata", False)
    metadata_tensor = None
    # Metadata ko encode karo, jaisa training mein kiya tha
    if use_metadata:
        metadata_tensor = encode_metadata(age, sex, localization)
    t2 = time.perf_counter()
    timings_ms["preprocess"] = (t2 - t1) * 1000

    with model_lock:
        # Single forward pass through the image encoder, shared by the OOD
        # guard (image branch only) and — if accepted — the classifier and
        # Grad-CAM below. explain=True needs gradients for the Grad-CAM
        # backward pass later, so only the explain=False path can use
        # inference_mode here.
        if explain:
            image_features = model.forward_features(image_tensor)
        else:
            with torch.inference_mode():
                image_features = model.forward_features(image_tensor)

        ood_distance = mahalanobis_distance(image_features)
        t3 = time.perf_counter()
        timings_ms["forward_and_backward"] = (t3 - t2) * 1000

        if ood_distance > OOD_THRESHOLD:
            timings_ms["heatmap_render"] = 0.0
            response_payload = {
                "status": "rejected",
                "message": REJECTION_MESSAGE,
                "ood": {"distance": ood_distance, "threshold": OOD_THRESHOLD},
            }
            t4 = time.perf_counter()
            timings_ms["serialize"] = (t4 - t3) * 1000
            response_payload["timings_ms"] = {k: round(v, 3) for k, v in timings_ms.items()}
            print(f"[predict] status=rejected ood_distance={ood_distance:.1f} threshold={OOD_THRESHOLD:.1f}")
            return JSONResponse(response_payload)

        gradcam_overlay_b64 = None
        if explain:
            heatmap, pred_class, probs = gradcam.generate_from_features(image_features, metadata_tensor)
            t3b = time.perf_counter()
            timings_ms["forward_and_backward"] += (t3b - t3) * 1000
            overlayed = overlay_heatmap(heatmap, pil_image)
            t4 = time.perf_counter()
            timings_ms["heatmap_render"] = (t4 - t3b) * 1000
            gradcam_overlay_b64 = image_to_base64(overlayed)
        else:
            with torch.inference_mode():
                if use_metadata:
                    meta_features = model.metadata_processor(metadata_tensor)
                    fused_features = torch.cat((image_features, meta_features), dim=1)
                    output = model.classifier_head(fused_features)
                else:
                    output = model.classifier_head(image_features)
                # argmax on raw logits — TEMPERATURE only reshapes the
                # displayed distribution, it can't flip the top-1 prediction.
                pred_class = output.argmax(dim=1).item()
                probs = (output / TEMPERATURE).softmax(dim=1)[0]
            t3b = time.perf_counter()
            timings_ms["forward_and_backward"] += (t3b - t3) * 1000
            timings_ms["heatmap_render"] = 0.0
            t4 = t3b

    prediction_set = conformal_prediction_set(probs)
    malignant_probability = float(sum(probs[i].item() for i in MALIGNANT_IDX))

    response_payload = {
        "status": "ok",
        "prediction": DX_LABELS[pred_class],
        "prediction_full_name": DX_FULL_NAMES[DX_LABELS[pred_class]],
        "confidence": float(probs[pred_class]),
        "all_probabilities": {DX_LABELS[i]: float(p) for i, p in enumerate(probs)},
        "prediction_set": prediction_set,
        "malignant": {
            "probability": malignant_probability,
            "flagged": malignant_probability >= MALIGNANT_THRESHOLD,
            "operating_point": MALIGNANT_OPERATING_POINT,
        },
        "ood": {"distance": ood_distance, "threshold": OOD_THRESHOLD},
    }
    if gradcam_overlay_b64 is not None:
        response_payload["gradcam_overlay_base64"] = gradcam_overlay_b64

    t5 = time.perf_counter()
    timings_ms["serialize"] = (t5 - t4) * 1000
    response_payload["timings_ms"] = {k: round(v, 3) for k, v in timings_ms.items()}

    print(
        f"[predict] status=ok explain={explain} prediction={DX_LABELS[pred_class]} "
        f"timings_ms={response_payload['timings_ms']}"
    )

    return JSONResponse(response_payload)
