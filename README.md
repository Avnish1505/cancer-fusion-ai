# 🩺 Cancer Fusion AI

> An open-source framework for multimodal medical AI, in active development.

⭐ If this project helps you, please consider starring the repository.
Contributions are welcome!

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blue)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)

> The Docker image (`Dockerfile:7`) targets Python 3.12; `runtime.txt` (used by the Procfile/buildpack deploy path) pins Python 3.10 — the two deploy paths currently disagree on version. A `LICENSE` file has not been added to this repo yet, so treat the MIT badge as aspirational (see [Known Limitations](#-known-limitations)).

Research-grade **open-source** multimodal framework for skin cancer classification using deep learning, metadata fusion, and explainable AI.

A multimodal deep learning pipeline for skin lesion classification on the **HAM10000** dataset — combining a CNN image encoder with patient metadata (age, sex, lesion location), and adding a **Grad-CAM explainability layer** so predictions aren't a pure black box. (LLM-generated clinical-report summaries are designed but not implemented — see [Roadmap](#-roadmap--not-yet-implemented).)

Built as an end-to-end, phased project: from an image-only baseline to a fused multimodal model, with an explainability layer on top.

**Topics:** `medical-ai` `pytorch` `deep-learning` `skin-cancer` `multimodal` `computer-vision` `gradcam` `healthcare-ai` `machine-learning` `ai` `research` `open-source`

---

## 🧠 Why this project

Skin cancer classifiers are usually judged only on accuracy. In a clinical-adjacent setting, that's not enough — a model needs to show **what** it looked at and **why**, and it needs to fold in the kind of context (age, sex, lesion site) that a real clinician would use. This project treats explainability as a first-class feature, not an afterthought.

---

## ❤️ Why Open Source?

Cancer Fusion AI is developed as an open-source project to make reproducible medical AI workflows accessible to students, researchers, and developers worldwide. The goal is to encourage collaboration, transparency, reproducible research, and community-driven improvements.

---

## 🏗️ Architecture

**Phase 1 — Image-only baseline**
ResNet50 as the feature extractor, fine-tuned on HAM10000's 7 diagnostic classes. The code supports ImageNet-pretrained initialization (`src/model.py:34-39`), but the shipped config now sets `pretrained: false` (`configs/config.yaml:20`) — the served checkpoint fully covers the model's `state_dict`, so pretrained weights would only be downloaded and then immediately overwritten. Pretrained init is still there for anyone training a fresh model from scratch.

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

**Phase 3 — Explainability layer (partially implemented)**
- ✅ **Grad-CAM** on the final conv block of the ResNet50 encoder (`app.py:103-131`), producing a heatmap over the exact regions that drove each prediction. Exposed via `/predict`'s `explain` query param — see [Known Limitations](#-known-limitations).
- 🚧 **LLM-generated clinical reports** — designed, not implemented. See [Roadmap](#-roadmap--not-yet-implemented).

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

The dataset is heavily imbalanced (`nv` ≈ 67% of samples, measured from `data/HAM10000_metadata.csv`) — handled via inverse-frequency class weighting (`src/dataset.py:252-266`, used in `src/train.py:126-129`).

**Split strategy:** stratified split **by `lesion_id`**, not by image, since some lesions have multiple photos. This prevents the same lesion's images from leaking across train/val/test (`src/dataset.py:209-225`).

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
  pretrained: false          # true only makes sense if you don't have a checkpoint yet
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

> ⚠️ The `configs/config.yaml` shipped in this repo currently has **two top-level `train:` blocks**. YAML does not error on duplicate top-level keys — it silently keeps only the last one, so the first block's values are dead. This should be cleaned up; until it is, only the second `train:` block (the one matching the example above) actually takes effect.

### Train

```bash
python -m src.train --config configs/config.yaml
```

Trains with early stopping (patience-based on val macro-F1), saves the best checkpoint to `paths.checkpoint_dir/best_model.pt` (`./checkpoints/best_model.pt` by default, `configs/config.yaml:36`).

### Evaluate

```bash
python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

`--checkpoint` is required (`src/evaluate.py:146`) — running the command without it will fail with an argparse error.

### Running the inference API

`app.py` serves the trained model over HTTP: `GET /health`, `GET /readyz`, and `POST /predict`. It loads `models/best_model.pt` (`app.py:80`), which is a **separately-trained, ready-to-serve checkpoint tracked with Git LFS** — it is *not* the same file `python -m src.train` produces above (that goes to `./checkpoints/best_model.pt`, a different path). Before running the API from a fresh clone:

```bash
git lfs install
git lfs pull
```

Without this, `models/best_model.pt` is left as an LFS pointer stub and the API will fail to start when it tries to load it (`app.py:80`).

Then either run it directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

or via Docker — note the checkpoint is **volume-mounted, not baked into the image** (see [Known Limitations](#-known-limitations)):

```bash
docker compose up
```

---

## 📈 Results

Phase 2 (image + metadata fusion), ResNet50 backbone.

| Metric | Value | Reproducibility |
|---|---|---|
| Best validation macro-F1 | **0.726** | Backed by the shipped checkpoint itself — `models/best_model.pt`'s stored `best_val_metric` field is `0.7261930300561489`, and its `epoch` field is `25`. Reproducible by loading that file, no retraining needed. |
| Train accuracy (best epoch) | ~0.90 | **Not currently reproducible from this repo.** `src/train.py:26-64` (`run_one_epoch`) computes this every epoch via `sklearn.metrics.accuracy_score`, but no results/log file is committed anywhere — this number reflects a past training run, not an artifact you can check. |
| Val accuracy (best epoch) | ~0.83 | Same caveat as above. |
| "Early-stopped at epoch 35/50" | — | **Not independently logged.** The checkpoint's own `epoch=25` (its best epoch) plus the configured `early_stopping_patience: 10` (`configs/config.yaml`) makes stopping at epoch 35 *consistent*, but no artifact records the actual stop point or total epochs run. |

> The train/val accuracy figures and the overfitting narrative below were true of one training run and are kept here for context, not as reproducible benchmarks.

Train/val F1 gap after ~epoch 25 indicates the model starts overfitting past that point — early stopping catches the best generalizing checkpoint rather than the lowest training loss. Class-imbalance handling (weighted loss) and further augmentation are natural next steps to close this gap further.

---

## 🔍 Explainability in action

Each prediction is paired with a **Grad-CAM heatmap** localizing the image regions the model relied on (`app.py:103-131`).

LLM-generated report summaries contextualizing a prediction in plain language are **designed but not implemented** — see [Roadmap](#-roadmap--not-yet-implemented). An earlier version of this README included an illustrative example of what such a report might say; it was not real model output and has been removed to avoid implying working code that doesn't exist.

---

## 🗂️ Project structure

```
cancer-fusion-ai/
├── app.py                # FastAPI service: /health, /readyz, /predict (Grad-CAM + explain toggle)
├── Dockerfile
├── docker-compose.yml     # mounts models/ and data/ as volumes — see Known Limitations
├── Procfile
├── src/
│   ├── config.py          # YAML config loader + validation
│   ├── dataset.py         # HAM10000Dataset, stratified split, class weights
│   ├── model.py            # CancerImageClassifier (image + metadata fusion)
│   ├── transforms.py       # train/eval augmentation pipelines
│   ├── train.py             # training loop, early stopping, checkpointing
│   ├── evaluate.py          # held-out evaluation
│   └── utils.py              # seeding, device selection, checkpoint I/O
├── configs/
│   └── config.yaml
├── models/
│   └── best_model.pt      # Git LFS — see "Running the inference API" above
├── data/
│   └── HAM10000_metadata.csv
├── tests/
├── requirements.txt
└── README.md
```

> `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, and `notebooks/explainability_demo.ipynb` were referenced in an earlier version of this README and this tree, but none of them exist in this repo (working tree or git history) — see [Roadmap](#-roadmap--not-yet-implemented).

---

## 🛠️ Tech stack

`PyTorch` · `torchvision` (ResNet50) · `scikit-learn` (splitting, metrics) · `OpenCV` (Grad-CAM overlay) · `FastAPI` / `uvicorn` (serving) · `pandas` / `NumPy`

---

## 📶 Performance

Measured locally — **not production-deployment numbers**: single uvicorn worker, CPU only, one machine, one request at a time, warm process (post cold-start), 5 runs each, same input image.

| Phase | `/predict` (`explain=true`, default) | `/predict?explain=false` |
|---|---|---|
| image_decode | ~0.4ms | ~0.4ms |
| preprocess | ~1.0ms | ~1.0ms |
| forward_and_backward | ~95-98ms | ~29-33ms |
| heatmap_render | ~1.1-1.2ms | 0ms (skipped) |
| serialize | ~0.8-0.9ms | ~0.02-0.06ms |
| **approx. total** | **~100ms** | **~31ms** |

Cold start (process import to first servable request), single worker: **~9.8s**.

These numbers characterize *where* per-request time goes on one local run, not what to expect under concurrent production traffic, on the actual deployment target, or with a GPU.

---

## ⚠️ Known Limitations

- **Cold start ≈ 9.8s locally (single worker, CPU).** Dominated by boot-time work in `app.py` at import: loading the full metadata CSV (`app.py:56`), recomputing the stratified train/val/test split (`app.py:57-67`) and re-fitting the age `StandardScaler` (`app.py:68-69`) on every process start, then building the model and loading the ~289MB checkpoint (`app.py:79-82`). None of this is cached across restarts.
- **Grad-CAM's backward pass is the dominant per-request cost.** `/predict` (Grad-CAM on, default) runs a full forward + backward pass plus heatmap rendering; `/predict?explain=false` skips all of that and runs a plain forward pass under `torch.inference_mode()`. Measured locally, that's the difference between ~100ms and ~31ms per request (see Performance above). Use `explain=false` when only the classification result is needed.
- **The shipped checkpoint (`models/best_model.pt`) is Git LFS-tracked and is *not* copied into the Docker image.** `Dockerfile` only creates an empty `models/` directory; the real checkpoint arrives via the `docker-compose.yml` volume mount. A container built and run without that mount (or without `git lfs pull` beforehand in whatever deploy path supplies the file) will fail at startup when it tries to load `models/best_model.pt`.
- **Training and serving disagree on checkpoint location.** `python -m src.train` writes to `paths.checkpoint_dir` (`./checkpoints/best_model.pt` by default, `configs/config.yaml:36`); `app.py:80` hardcodes `models/best_model.pt`. Retraining locally does not automatically update what the API serves — you'd need to copy the new checkpoint over.
- **`configs/config.yaml` has two top-level `train:` blocks.** YAML silently keeps only the last one (PyYAML doesn't error on duplicate keys), so the first block's values are dead code in the file. Not a runtime bug today (the second block happens to be the intended one), but misleading to read.
- **LLM-generated clinical report summaries are not implemented.** See [Roadmap](#-roadmap--not-yet-implemented).

---

## 🚧 Roadmap / Not Yet Implemented

Product roadmap:
- [ ] **Phase 4** — Model evaluation deep-dive: per-class precision/recall/confusion matrix, error analysis on misclassified melanoma cases (highest clinical stakes)
- [ ] **Phase 5** — Lightweight deployment: Gradio/Streamlit demo for interactive upload → prediction → Grad-CAM → report
- [ ] **LLM-generated clinical report summaries** — designed, not implemented. No NVIDIA NIM / OpenAI code exists anywhere in this repo's working tree or git history; `openai` sits unused in `requirements.txt`. A previous README version showed an illustrative example report as if it were real output — it wasn't, and has been removed.
- [ ] **Explainability demo notebook** (`notebooks/explainability_demo.ipynb`) — referenced in earlier docs, never actually committed to this repo.

Repo housekeeping (referenced elsewhere in this README, not yet added):
- [ ] `LICENSE` — the MIT badge above is aspirational until this exists
- [ ] `CONTRIBUTING.md`
- [ ] `CODE_OF_CONDUCT.md`
- [ ] `SECURITY.md`
- [ ] `CHANGELOG.md`

Longer-term:
- [x] Docker deployment — implemented (`Dockerfile`, `docker-compose.yml`, `Procfile`)
- [x] FastAPI APIs — implemented (`app.py`: `/health`, `/readyz`, `/predict`)
- [ ] CI/CD
- [ ] Hugging Face demo
- [ ] ONNX export
- [ ] Mobile inference
- [ ] Better explainability
- [ ] Benchmarking

---

## 🤝 Contributing

Contributions are welcome! You can contribute by:

- Improving documentation
- Reporting bugs
- Fixing issues
- Improving model performance
- Adding explainability techniques
- Improving deployment

Please open an Issue before submitting major Pull Requests. This repo doesn't yet have `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` / `CHANGELOG.md` (see [Roadmap](#-roadmap--not-yet-implemented)) — for now, an Issue is the best way to coordinate before a big change.

---

## 📄 License

This project intends to use the MIT License, but **no `LICENSE` file has been added to this repo yet** (see [Roadmap](#-roadmap--not-yet-implemented)). Treat the badge at the top of this README as aspirational until that's fixed.

---

## ⚠️ Disclaimer

This is a research and educational project. It is **not** a certified medical device and must **not** be used for real clinical diagnosis or treatment decisions. All outputs should be verified by a qualified healthcare professional.

---

## ⭐ Support

If you find this repository useful:

- ⭐ Star the repository
- 🍴 Fork it
- 🐛 Report issues
- 💡 Suggest new features

Your support helps this project reach more students, researchers, and developers.

---

## 👤 Author & Maintainer

**Avnish Singh** — AI/ML Engineer, B.Tech CSE, Babu Banarasi Das University
Primary Maintainer of Cancer Fusion AI · Co-author, ADG 2026 International Conference (AI in Legal Technology)

[LinkedIn](https://www.linkedin.com/in/avnish-singh-a94772309) · [Portfolio](https://website-bzc5.vercel.app/)
