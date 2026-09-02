# Experiment with your own data

The pipeline isn't hardcoded to kidney CT scans — it's a generic image classifier. This is for training it on your own images and classes, tweaking it, or updating the public demo with a new model. Assumes you've already done the [local install](RUN_LOCALLY.md#1-install).

## Train on your own images

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
3. Train and evaluate exactly as with the kidney dataset:
   ```bash
   python src/train.py --data-dir data/raw --epochs 10
   python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw
   python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
   ```

Classes and their names are picked up automatically from your folder names — nothing else to configure. See [`src/model.py`](../src/model.py) if you want to swap in a different backbone than ResNet18.

## Update the free live demo

After retraining, push the new model to the public browser demo:

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

If your new model has different classes than Cyst/Normal/Stone/Tumor, also update the `CLASSES` array and page text in `hf_space_static/index.html` to match.

---
Back: [Run locally](RUN_LOCALLY.md) · [How it works](HOW_IT_WORKS.md) · [Back to README](../README.md)
