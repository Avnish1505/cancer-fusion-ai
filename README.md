# Cancer Fusion AI — Multi-Modal Skin Lesion Classification

**Phase 1: Image-only baseline** (Phase 2 will add patient metadata fusion, Phase 3 explainability)

## What this is

A research-grade pipeline for skin cancer classification on the HAM10000 dataset,
built to eventually fuse image + patient metadata + generate human-readable
explanation reports — not just a "cancer / no cancer" black box.

This is Phase 1: a solid, properly-validated, leak-free image classification
baseline. Everything downstream (fusion, explainability) builds on this foundation.

## Why this differs from typical student projects

- **Lesion-level train/val/test splitting** — HAM10000 has multiple photos of the
  same lesion. Splitting by `image_id` naively causes data leakage (same lesion's
  photos in both train and test), which inflates accuracy artificially. This
  pipeline splits by `lesion_id` instead. Verified by an explicit leakage test.
- **Class-weighted loss** — HAM10000 is ~67% one class (`nv`). Naive training
  collapses to predicting the majority class. Class weights correct for this.
- **Melanoma recall tracked explicitly** — in medical AI, overall accuracy is a
  misleading headline metric. A missed melanoma (false negative) is far more
  costly than a false alarm, so this is reported separately in evaluation.
- **Fails loudly and clearly, not silently** — missing images, corrupt files, bad
  configs, and malformed data all raise clear, actionable errors instead of
  cryptic crashes mid-training or (worse) silently corrupting results.

## Project structure

```
cancer-fusion-ai/
├── configs/
│   └── config.yaml          # All settings — nothing hardcoded in code
├── src/
│   ├── config.py             # Config loading + validation
│   ├── dataset.py            # HAM10000 dataset, leak-free splitting, class weights
│   ├── transforms.py         # Augmentation pipeline
│   ├── model.py               # ResNet50/EfficientNet-B0 classifier
│   ├── train.py               # Training loop with early stopping + checkpointing
│   ├── evaluate.py            # Confusion matrix, classification report, melanoma recall
│   └── utils.py                # Seeding, device handling, checkpoint I/O
├── tests/
│   ├── smoke_test.py          # End-to-end pipeline test on fake data
│   └── edge_case_test.py      # Error-handling tests (missing files, bad configs, etc.)
└── requirements.txt
```

## Setup — run this on Kaggle (recommended, free GPU + dataset already hosted)

1. Create a new Kaggle Notebook.
2. Add the dataset: **"Skin Cancer MNIST: HAM10000"** (search it in Add Data).
3. Upload this repo's `src/` and `configs/` folders (or `git clone` if you push
   this to GitHub first — recommended, since that's your actual goal).
4. In the notebook:
   ```bash
   pip install -r requirements.txt
   ```
5. Update `configs/config.yaml` — check the exact dataset paths Kaggle mounts
   (usually under `/kaggle/input/skin-cancer-mnist-ham10000/`, but verify with
   `!ls /kaggle/input/`).
6. Enable GPU: Notebook settings → Accelerator → GPU T4 x2 (or similar).
7. Run:
   ```bash
   python -m src.train --config configs/config.yaml
   python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
   ```

## Setup — run locally first (to verify everything works before using GPU quota)

```bash
pip install -r requirements.txt
python tests/smoke_test.py       # ~30 seconds, verifies the full pipeline on fake data
python tests/edge_case_test.py   # verifies error handling
```

Both should print `ALL ... TESTS PASSED ✅`. **Run these before every real
training run on Kaggle** — catching a bug in a 10-second local test beats
discovering it 15 minutes into a GPU run.

## Roadmap

- [x] **Phase 1** — Image-only baseline (this repo state)
- [ ] **Phase 2** — Fuse patient metadata (age, sex, lesion location) via a
      tabular branch + fusion layer
- [ ] **Phase 3** — Explainability: Grad-CAM visual heatmaps + LLM-generated
      plain-language diagnostic reasoning
- [ ] **Phase 4** — FastAPI serving + simple demo UI
- [ ] **Phase 5** — Polish, benchmark comparison table, GitHub release

## Dataset citation

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. *Sci Data* 5, 180161 (2018).

## Related work

*(Fill this in as you read papers — write each entry in your own words,
summarizing their approach and how this project's angle differs. This
section is what makes the repo look like real research rather than a copy.)*

## Disclaimer

This is a research/educational project, not a medical device. It is not
validated for clinical use and must not be used for actual diagnosis.
