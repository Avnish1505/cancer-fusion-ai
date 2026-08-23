import base64
import io
import os
import threading
import time

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
load_checkpoint("models/best_model.pt", model, device=device)
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

# Inference-time display calibration only — does NOT affect which class
# wins (argmax is invariant to dividing all logits by a positive scalar)
# and does NOT touch training/the checkpoint. The served model is
# genuinely overconfident (NV is ~67% of HAM10000's training data), so
# raw logits routinely saturate softmax to ~100%/0%. T>1 softens the
# displayed probability distribution without changing the prediction.
TEMPERATURE = 2.5


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

    def generate(self, image_tensor, metadata_tensor, class_idx=None):
        output = self.model(image_tensor, metadata_tensor)
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

    gradcam_overlay_b64 = None
    if explain:
        # Identical compute path to before — no no_grad/inference_mode wrapping,
        # only the lock (see model_lock comment above) added around it.
        with model_lock:
            heatmap, pred_class, probs = gradcam.generate(image_tensor, metadata_tensor)
        t3 = time.perf_counter()
        timings_ms["forward_and_backward"] = (t3 - t2) * 1000

        overlayed = overlay_heatmap(heatmap, pil_image)
        t4 = time.perf_counter()
        timings_ms["heatmap_render"] = (t4 - t3) * 1000

        gradcam_overlay_b64 = image_to_base64(overlayed)
    else:
        # Prediction-only path: skip zero_grad/backward/heatmap entirely.
        with model_lock:
            with torch.inference_mode():
                output = model(image_tensor, metadata_tensor)
                # argmax on raw logits — TEMPERATURE only reshapes the
                # displayed distribution, it can't flip the top-1 prediction.
                pred_class = output.argmax(dim=1).item()
                probs = (output / TEMPERATURE).softmax(dim=1)[0]
        t3 = time.perf_counter()
        timings_ms["forward_and_backward"] = (t3 - t2) * 1000
        timings_ms["heatmap_render"] = 0.0
        t4 = t3

    response_payload = {
        "prediction": DX_LABELS[pred_class],
        "prediction_full_name": DX_FULL_NAMES[DX_LABELS[pred_class]],
        "confidence": float(probs[pred_class]),
        "all_probabilities": {DX_LABELS[i]: float(p) for i, p in enumerate(probs)},
    }
    if gradcam_overlay_b64 is not None:
        response_payload["gradcam_overlay_base64"] = gradcam_overlay_b64

    t5 = time.perf_counter()
    timings_ms["serialize"] = (t5 - t4) * 1000
    response_payload["timings_ms"] = {k: round(v, 3) for k, v in timings_ms.items()}

    print(
        f"[predict] explain={explain} prediction={DX_LABELS[pred_class]} "
        f"timings_ms={response_payload['timings_ms']}"
    )

    return JSONResponse(response_payload)