# How it works

A technical explanation of what's actually in this repo.

## The task

Classify a single kidney CT scan slice into one of four categories: **Cyst, Normal, Stone, or Tumor**. Framed as ordinary 4-class image classification.

## The data

[mhmad240/kidney-ct-classification](https://huggingface.co/datasets/mhmad240/kidney-ct-classification) on Hugging Face — 8,712+ labeled kidney CT images with train/validation/test splits, sourced from hospital CT scans. This repo trains on a balanced subset (1,000 train / 240 validation / 240 test images) fetched via `data/fetch_subset.py`, rather than the full set, to keep training fast on a laptop CPU.

## The model

A ResNet18 convolutional network, pretrained on ImageNet and fine-tuned on the kidney CT images (`src/model.py`, `src/train.py`) — a standard transfer-learning setup: reuse the general visual features ResNet18 already learned from millions of everyday photos, and retrain only enough of it to recognize the four kidney-scan categories. Small and fast enough to train on a CPU in a few minutes; a larger backbone (EfficientNet, a Vision Transformer) would likely score higher but take longer to train.

## Explainability: Grad-CAM

The local app (not the free browser demo — see below) also produces a Grad-CAM heatmap (`src/gradcam.py`) over the input image, highlighting which pixels most influenced the prediction. This doesn't explain *why* in a clinical sense, but it's the minimum needed to avoid a pure black-box output — you can at least see whether the model is looking at the kidney or, say, an unrelated corner of the scan.

## Results

Trained for 6 epochs on the 1,000-image training subset, evaluated on the held-out 240-image test set (no overlap with train/validation):

| Class  | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Cyst   | 0.97 | 0.98 | 0.98 |
| Normal | 0.98 | 0.98 | 0.98 |
| Stone  | 1.00 | 0.98 | 0.99 |
| Tumor  | 0.98 | 0.98 | 0.98 |

**Overall accuracy: 98%** · **Tumor-vs-rest ROC-AUC: 0.9998**

This is a single-source dataset, so these numbers show the pipeline and task are sound — not that the model generalizes to scans from other hospitals or machines. See [BACKGROUND.md](BACKGROUND.md) for what real clinical validation would require.

## Two versions of the demo

There are two separate apps in this repo, because Hugging Face changed its pricing partway through this project — free hosting for a Python backend (needed for Grad-CAM) now requires a paid PRO plan, so:

- **`app/demo_app.py`** — the full Gradio app, with the Grad-CAM heatmap. Runs locally (see [Run locally](RUN_LOCALLY.md)); would need a paid host to run publicly.
- **`hf_space_static/`** — a lighter version with no Grad-CAM, converted to [ONNX](https://onnx.ai/) and run entirely client-side in the visitor's browser via `onnxruntime-web`. No backend, no cost, nothing ever uploaded to a server. This is what's live at the [public demo link](https://huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo).

## Repo layout

```
data/
  download_data.py     # full dataset -> data/raw/ (needs the `datasets` package)
  fetch_subset.py       # fast balanced subset -> data/raw/ (no extra deps)
  prepare_own_data.py   # turn your own flat image folders into data/raw/
  sample/                 # 24 real CT images (6/class) for quick testing
src/
  dataset.py             # PyTorch Dataset + image transforms
  model.py               # ResNet18 transfer-learning model
  train.py               # training loop
  evaluate.py             # accuracy / confusion matrix / per-class metrics / ROC-AUC
  gradcam.py              # Grad-CAM explainability
app/
  demo_app.py             # Gradio web app: upload an image -> prediction + heatmap
hf_space_static/
  index.html               # the free, browser-only demo (ONNX Runtime Web, no backend)
  export_onnx.py            # checkpoint -> model.onnx, run after retraining to update the live demo
```

`data/raw/`, `checkpoints/`, and `hf_space_static/model.onnx` are gitignored — they're generated locally by the commands in [Run locally](RUN_LOCALLY.md), not stored in the repo.

---
Next: [Run it locally](RUN_LOCALLY.md) · [Experiment with your own data](EXPERIMENT.md) · [Background & industry context](BACKGROUND.md) · [Back to README](../README.md)
