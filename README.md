# Cancer Fusion AI 🔬

A multimodal deep learning pipeline for skin lesion classification on the **HAM10000** dataset — combining a CNN image encoder with patient metadata (age, sex, lesion location), and adding an **explainability layer** (Grad-CAM + LLM-generated clinical reports) so predictions aren't a black box.

Built as an end-to-end, phased project: from an image-only baseline to a fused multimodal model to human-readable AI explanations.

---

## 🧠 Why this project

Skin cancer classifiers are usually judged only on accuracy. In a clinical-adjacent setting, that's not enough — a model needs to show **what** it looked at and **why**, and it needs to fold in the kind of context (age, sex, lesion site) that a real clinician would use. This project treats explainability as a first-class feature, not an afterthought.

---

## 🏗️ Architecture

**Phase 1 — Image-only baseline**
ResNet50 (ImageNet pretrained) as the feature extractor, fine-tuned on HAM10000's 7 diagnostic classes.

**Phase 2 — Metadata fusion**
Patient metadata (age, sex, anatomical site) is processed through a small MLP and concatenated with the image embedding before the final classifier head — so the model reasons over pixels *and* clinical context together.

```
Image ──► ResNet50 (encoder) ──► image_features (2048-d)
                                            │
Metadata ──► MLP (128 → 64) ──► meta_features (64-d)
                                            │
                                    concat + classifier head
                                            │
                                      7-class logits
```

**Phase 3 — Explainability layer** ✅ *(current)*
- **Grad-CAM** on the final conv block of the ResNet50 encoder, producing a heatmap over the exact regions that drove each prediction.
- **LLM-generated clinical reports**: the prediction, confidence, Grad-CAM context, and patient metadata are passed to an LLM (via NVIDIA NIM, OpenAI-compatible API) which writes a short, readable explanation — always with an explicit disclaimer that this is a research prototype, not a diagnostic tool.

---

## 📊 Dataset

[HAM10000 ("Human Against Machine")](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — ~10,000 dermatoscopic images across 7 classes:

| Code | Diagnosis |
|------|-----------|
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

The dataset is heavily imbalanced (`nv` ≈ 67% of samples) — handled via inverse-frequency class weighting in the loss function.

**Split strategy:** stratified split **by `lesion_id`**, not by image, since some lesions have multiple photos. This prevents the same lesion's images from leaking across train/val/test.

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/cancer-fusion-ai.git
cd cancer-fusion-ai
pip install -r requirements.txt
```

Download the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and update the paths in `configs/config.yaml`:

```yaml
data:
  metadata_csv: "path/to/HAM10000_metadata.csv"
  images_dir_part1: "path/to/HAM10000_images_part_1"
  images_dir_part2: "path/to/HAM10000_images_part_2"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  backbone: "resnet50"      # or "efficientnet_b0"
  num_classes: 7
  pretrained: true
  dropout: 0.3

train:
  use_metadata: true
  image_size: 224
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  early_stopping_patience: 10
  use_class_weights: true
  device: "cuda"
```

### Train

```bash
python -m src.train --config configs/config.yaml
```

Trains with early stopping (patience-based on val macro-F1), saves the best checkpoint to `paths.checkpoint_dir/best_model.pt`.

### Evaluate

```bash
python -m src.evaluate --config configs/config.yaml
```

---

## 📈 Results

Phase 2 (image + metadata fusion), ResNet50 backbone, early-stopped at epoch 35/50:

| Metric | Value |
|---|---|
| Best validation macro-F1 | **0.726** |
| Train accuracy (best epoch) | ~0.90 |
| Val accuracy (best epoch) | ~0.83 |

> Train/val F1 gap after ~epoch 25 indicates the model starts overfitting past that point — early stopping catches the best generalizing checkpoint rather than the lowest training loss. Class-imbalance handling (weighted loss) and further augmentation are natural next steps to close this gap further.

---

## 🔍 Explainability in action

Each prediction is paired with:
1. A **Grad-CAM heatmap** localizing the image regions the model relied on
2. An **LLM-generated report** contextualizing the prediction with lesion morphology and patient metadata, always flagged as research-only

Example (melanoma sample, val set):

> *"The AI model analyzed the image of the 45-year-old female patient's lower extremity lesion and predicted melanoma with 100.0% confidence... The model's Grad-CAM heatmap specifically focused on the most visually heterogeneous and structurally irregular regions of the lesion to justify its high-confidence prediction. It is crucial to emphasize that this AI system is a research and educational prototype and is NOT a medical diagnostic tool..."*

The heatmap correctly localizes onto the asymmetric, irregularly pigmented core of the lesion — the same region a dermatologist would visually flag under the ABCDE rule.

---

## 🗂️ Project structure

```
cancer-fusion-ai/
├── src/
│   ├── config.py        # YAML config loader + validation
│   ├── dataset.py        # HAM10000Dataset, stratified split, class weights
│   ├── model.py           # CancerImageClassifier (image + metadata fusion)
│   ├── transforms.py      # train/eval augmentation pipelines
│   ├── train.py            # training loop, early stopping, checkpointing
│   ├── evaluate.py         # held-out evaluation
│   └── utils.py             # seeding, device selection, checkpoint I/O
├── configs/
│   └── config.yaml
├── notebooks/
│   └── explainability_demo.ipynb   # Grad-CAM + LLM report generation (Phase 3)
└── requirements.txt
```

---

## 🛠️ Tech stack

`PyTorch` · `torchvision` (ResNet50) · `scikit-learn` (splitting, metrics) · `OpenCV` (Grad-CAM overlay) · `NVIDIA NIM` (LLM inference, OpenAI-compatible API) · `pandas` / `NumPy`

---

## 🚧 Roadmap

- [ ] **Phase 4** — Model evaluation deep-dive: per-class precision/recall/confusion matrix, error analysis on misclassified melanoma cases (highest clinical stakes)
- [ ] **Phase 5** — Lightweight deployment: Gradio/Streamlit demo for interactive upload → prediction → Grad-CAM → report

---

## ⚠️ Disclaimer

This is a research and educational project. It is **not** a certified medical device and must **not** be used for real clinical diagnosis or treatment decisions. All outputs should be verified by a qualified healthcare professional.

---

## 👤 Author

**Avnish Singh** — B.Tech CSE, Babu Banarasi Das University
Co-author, ADG 2026 International Conference (AI in Legal Technology)
[www.linkedin.com/in/avnish-singh-a94772309] · [https://website-bzc5.vercel.app/]
