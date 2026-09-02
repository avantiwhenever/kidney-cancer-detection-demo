# Developer notes

Deeper technical detail that didn't need to be in the main [README](../README.md).

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

`data/raw/`, `checkpoints/`, and `hf_space_static/model.onnx` are gitignored — they're generated locally, not stored in the repo.

## Full results

ResNet18 trained for 6 epochs on a balanced 1,000-image subset, evaluated on a held-out 240-image test set:

| Class  | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Cyst   | 0.97 | 0.98 | 0.98 |
| Normal | 0.98 | 0.98 | 0.98 |
| Stone  | 1.00 | 0.98 | 0.99 |
| Tumor  | 0.98 | 0.98 | 0.98 |

Overall accuracy: 98%. Tumor-vs-rest ROC-AUC: 0.9998. This is on a single-source dataset, so it shows the pipeline and task work well — not that it generalizes to other hospitals/scanners. See [BACKGROUND.md](BACKGROUND.md) for what real clinical validation would require.

## Updating the free live demo

After retraining, refresh the hosted version:

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

Requires `pip install huggingface_hub` and `hf auth login` with a write token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
