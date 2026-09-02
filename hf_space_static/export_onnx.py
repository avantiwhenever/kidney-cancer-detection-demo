"""Export a trained checkpoint to ONNX for the browser-only static demo.

Run this after src/train.py, then re-upload hf_space_static/ to the Hugging
Face Space to update the live demo.

Usage:
    python hf_space_static/export_onnx.py --checkpoint checkpoints/resnet18_kidney.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_kidney.pt")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "model.onnx"))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    print("classes (must match CLASSES in index.html):", classes)

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
