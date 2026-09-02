"""Gradio demo: upload a kidney CT slice, get a predicted class, per-class
probabilities, and a Grad-CAM overlay showing what the model attended to.

Research/educational demo only -- see DISCLAIMER.md. Not a diagnostic device.

Usage:
    python app/demo_app.py --checkpoint checkpoints/resnet18_kidney.pt
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import get_transforms  # noqa: E402
from gradcam import GradCAM  # noqa: E402
from model import build_model  # noqa: E402

DISCLAIMER = (
    "Research demo only. Not a medical device and not validated for "
    "clinical use. Do not use for real diagnosis."
)


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes, device


def make_predictor(checkpoint_path):
    model, classes, device = load_model(checkpoint_path)
    cam = GradCAM(model, model.layer4[-1])
    transform = get_transforms(train=False)

    def predict(pil_image):
        image = pil_image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

        heatmap, _ = cam(tensor)
        heatmap_rgb = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
        base = np.array(image.resize(heatmap.shape[::-1]))
        overlay = (0.5 * base + 0.5 * heatmap_rgb).astype(np.uint8)

        return {c: float(p) for c, p in zip(classes, probs)}, Image.fromarray(overlay)

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_kidney.pt")
    args = parser.parse_args()

    predict = make_predictor(args.checkpoint)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Kidney CT slice"),
        outputs=[
            gr.Label(num_top_classes=4, label="Predicted class"),
            gr.Image(label="Grad-CAM overlay"),
        ],
        title="Kidney CT Classifier Demo",
        description=DISCLAIMER,
    )
    demo.launch()


if __name__ == "__main__":
    main()
