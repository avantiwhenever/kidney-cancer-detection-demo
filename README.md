# Kidney Cancer Detection — Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A free tool that looks at a kidney CT scan image and predicts whether it shows a **Cyst, a Tumor, a Stone, or a Normal kidney**. Built to be usable by anyone — no coding required to try it, and open-source for anyone who wants to look under the hood or build on it.

> ⚠️ **Not a medical device.** This is a research/education project, not a clinical tool. Do not use it for real diagnosis. See [DISCLAIMER.md](DISCLAIMER.md).

## Try it — free, in your browser, no install

**👉 [huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo](https://huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo)**

Click the link, then either upload a kidney CT image or click one of the four example pictures. It runs right there in your browser — nothing is uploaded to any server, no account needed.

## What it actually does

It was shown thousands of labeled kidney CT scan images ahead of time and learned the visual patterns that tell apart a cyst, a tumor, a kidney stone, and a healthy kidney. When you give it a new image, it compares what it sees against what it learned and gives you its best guess, along with how confident it is in each possibility.

In testing on images it hadn't seen before, it got the right answer about **98 out of 100 times**. That's a good sign the approach works — but it was only tested on images from one source, so it hasn't been proven to work reliably on scans from other hospitals or machines. That, plus a lot more clinical testing, is what separates a demo like this from something a doctor could actually rely on.

## For developers

Want to run it yourself, see how it was built, or train it on your own images? Everything below needs a terminal and Python — skip it if you just wanted to try the demo above.

```bash
git clone https://github.com/avantiwhenever/kidney-cancer-detection-demo.git
cd kidney-cancer-detection-demo

# CPU-only PyTorch (drop --index-url if you have a CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Get a balanced training set from Hugging Face, then train (a few minutes on a laptop CPU)
python data/fetch_subset.py --per-class 250 60 60
python src/train.py --data-dir data/raw --epochs 6

# Check how well it did
python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw

# Launch the full local app (includes a heatmap showing what the model focused on)
python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
```

The app opens at `http://127.0.0.1:7860`. Upload any kidney CT image (or one from `data/sample/`) to try it.

**Want to train it on your own images instead of kidney scans?** It's a generic image classifier — put your images in one folder per class, then:

```bash
python data/prepare_own_data.py --input-dir my_data --output-dir data/raw
python src/train.py --data-dir data/raw --epochs 10
```

Classes are picked up automatically from your folder names.

Full results table, repo layout, and how to update the live demo: [docs/DEVELOPERS.md](docs/DEVELOPERS.md). Regulatory/clinical context and what it'd take to turn this into a real medical product: [docs/BACKGROUND.md](docs/BACKGROUND.md).

## Dataset & license

- Code: **MIT** — see [LICENSE](LICENSE). Free to use, modify, and redistribute.
- Training data: [mhmad240/kidney-ct-classification](https://huggingface.co/datasets/mhmad240/kidney-ct-classification) on Hugging Face — check its page for license terms before redistribution or commercial use.

## Contributing

Issues and pull requests are welcome — better models, DICOM support, and external validation on other datasets are all good directions.
