# Run it locally

For running the full app on your own machine — including the Grad-CAM heatmap, which the free [online demo](https://huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo) doesn't have.

## 1. Install

Requires Python 3.9+.

```bash
git clone https://github.com/avantiwhenever/kidney-cancer-detection-demo.git
cd kidney-cancer-detection-demo

# CPU-only PyTorch (drop --index-url if you have a CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Get a trained model

The repo doesn't ship a trained checkpoint (models are build artifacts, not source), so train one — it only takes a few minutes on a laptop CPU:

```bash
# Get a balanced training set straight from Hugging Face (no extra setup needed)
python data/fetch_subset.py --per-class 250 60 60

# Train
python src/train.py --data-dir data/raw --epochs 6

# Check how well it did
python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw
```

## 3. Launch the app

```bash
python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
```

Opens at `http://127.0.0.1:7860`. Upload any kidney CT image — your own, or one from `data/sample/` — and you'll get the predicted class, per-class probabilities, and a Grad-CAM overlay showing what the model focused on.

---
Next: [Experiment with your own data](EXPERIMENT.md) · [How it works](HOW_IT_WORKS.md) · [Back to README](../README.md)
