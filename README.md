# Kidney Cancer Detection — Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source, end-to-end image classifier for kidney CT scans — **Cyst / Normal / Stone / Tumor** — with an explainability heatmap (Grad-CAM) and a simple web app anyone can use, on the included data or their own.

> **Not a medical device.** This is a research/education project, not a clinical tool. Do not use it for real diagnosis. See [DISCLAIMER.md](DISCLAIMER.md).

## 🔗 Try it now — free, no install

**[huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo](https://huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo)**

Runs entirely in your browser (ONNX Runtime Web) — click an example image or upload your own, no signup, no upload to any server. It's a lighter version of the app: same classifier, but no Grad-CAM heatmap (that needs a Python backend, which costs money to host — see `hf_space_static/` for how this was built and deployed). For the full app with Grad-CAM, run it locally with the steps below.

## Results

ResNet18 trained for 6 epochs on a balanced 1,000-image subset, evaluated on a held-out 240-image test set:

| Class  | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Cyst   | 0.97 | 0.98 | 0.98 |
| Normal | 0.98 | 0.98 | 0.98 |
| Stone  | 1.00 | 0.98 | 0.99 |
| Tumor  | 0.98 | 0.98 | 0.98 |

**Overall accuracy: 98%** · **Tumor-vs-rest ROC-AUC: 0.9998**

This is on a single-source dataset, so it shows the pipeline and task work well — not that it generalizes to other hospitals/scanners. See [docs/BACKGROUND.md](docs/BACKGROUND.md) for what real clinical validation would require.

## 1. Install

Requires Python 3.9+.

```bash
git clone https://github.com/avantiwhenever/kidney-cancer-detection-demo.git
cd kidney-cancer-detection-demo

# CPU-only PyTorch (drop --index-url if you have a CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Try the pretrained model

The repo doesn't ship a trained checkpoint (models are build artifacts, not source), so train one first — it only takes a couple of minutes on a laptop CPU:

```bash
# Get a balanced training set straight from Hugging Face (no extra setup needed)
python data/fetch_subset.py --per-class 250 60 60

# Train (few minutes on CPU)
python src/train.py --data-dir data/raw --epochs 6

# Check how well it did
python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw

# Launch the interactive app in your browser
python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
```

The app opens at `http://127.0.0.1:7860`. Upload a kidney CT image (or drag one in from `data/sample/`) and you'll get the predicted class, per-class probabilities, and a Grad-CAM overlay showing what the model focused on.

## 3. Use it on your own images

Once you have a trained checkpoint (from step 2), just point the app at your own picture:

```bash
python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
```

Then upload any kidney CT slice through the browser UI. No coding required beyond the one command above.

## 4. Train it on your own dataset

The pipeline isn't hardcoded to kidney CT scans — it's a generic image classifier. To train it on your own images and classes:

1. Organize your images into one folder per class:
   ```
   my_data/
     ClassA/*.jpg
     ClassB/*.jpg
     ClassC/*.jpg
   ```
2. Split them into train/validation/test automatically:
   ```bash
   python data/prepare_own_data.py --input-dir my_data --output-dir data/raw
   ```
3. Train and evaluate exactly as in step 2:
   ```bash
   python src/train.py --data-dir data/raw --epochs 10
   python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw
   python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
   ```

The number of classes and their names are picked up automatically from your folder names — nothing else to configure.

## What's in this repo

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

`data/raw/`, `checkpoints/`, and `hf_space_static/model.onnx` are gitignored — they're generated locally by the commands above, not stored in the repo.

### Updating the free live demo

After retraining, refresh the hosted version at the link above:

```bash
pip install -r hf_space_static/requirements-export.txt
python hf_space_static/export_onnx.py --checkpoint checkpoints/resnet18_kidney.pt
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id='avantiwhenever/kidney-cancer-detection-demo',
    repo_type='space',
    folder_path='hf_space_static',
)"
```

(Requires `pip install huggingface_hub` and `hf auth login` with a write token — see [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).)

## Dataset & license

- Model code: **MIT** — see [LICENSE](LICENSE). Free to use, modify, and redistribute.
- Training data: [mhmad240/kidney-ct-classification](https://huggingface.co/datasets/mhmad240/kidney-ct-classification) on Hugging Face. Check that dataset's own page for its license terms before redistribution or commercial use.

## Contributing

Issues and pull requests are welcome — better models, DICOM support, a hosted demo link, and external validation on other datasets are all good directions. See [docs/BACKGROUND.md](docs/BACKGROUND.md) for the fuller roadmap and context.
