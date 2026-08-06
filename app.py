import base64
import io

import cv2
import matplotlib
import numpy as np
import torch

import pandas as pd

matplotlib.use('Agg')
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from sklearn.preprocessing import StandardScaler
from src.config import load_config
from src.dataset import DX_FULL_NAMES, DX_LABELS
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint

app = FastAPI(title="Cancer Fusion AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vercel URL ke liye isko update karna padega
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Load model once at startup ----
config = load_config("configs/config.yaml")

# ---- Recreate metadata encoding exactly as training did ----
# NOTE: Yeh path aapke local setup ke hisaab se adjust karna pad sakta hai
metadata_csv = pd.read_csv("data/HAM10000_metadata.csv")
age_median = metadata_csv["age"].median()
metadata_csv["age"] = metadata_csv["age"].fillna(age_median)
age_scaler = StandardScaler()
age_scaler.fit(metadata_csv[["age"]])

sex_dummies = pd.get_dummies(metadata_csv["sex"].fillna("unknown"), prefix="sex", dtype=float)
loc_dummies = pd.get_dummies(metadata_csv["localization"].fillna("unknown"), prefix="loc", dtype=float)
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
        return heatmap, class_idx, output.softmax(dim=1)[0]


gradcam = FusionGradCAM(model, model.encoder.layer4[-1])


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


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(50),
    sex: str = Form("male"),
    localization: str = Form("back"),
):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    use_metadata = config["train"].get("use_metadata", False)
    metadata_tensor = None
    # Metadata ko encode karo, jaisa training mein kiya tha
    if use_metadata:
        metadata_tensor = encode_metadata(age, sex, localization)

    heatmap, pred_class, probs = gradcam.generate(image_tensor, metadata_tensor)
    overlayed = overlay_heatmap(heatmap, pil_image)

    return JSONResponse({
        "prediction": DX_LABELS[pred_class],
        "prediction_full_name": DX_FULL_NAMES[DX_LABELS[pred_class]],
        "confidence": float(probs[pred_class]),
        "all_probabilities": {DX_LABELS[i]: float(p) for i, p in enumerate(probs)},
        "gradcam_overlay_base64": image_to_base64(overlayed),
    })