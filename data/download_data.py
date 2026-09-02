"""Download the full mhmad240/kidney-ct-classification dataset from Hugging
Face and lay it out as data/raw/<split>/<class_name>/*.jpg for
torchvision.ImageFolder.

Usage:
    python data/download_data.py
"""
import os

from datasets import load_dataset

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")


def main():
    ds = load_dataset("mhmad240/kidney-ct-classification")
    for split, data in ds.items():
        for i, ex in enumerate(data):
            label_name = CLASSES[ex["label"]]
            out_path = os.path.join(OUT_DIR, split, label_name)
            os.makedirs(out_path, exist_ok=True)
            ex["image"].convert("RGB").save(os.path.join(out_path, f"{i}.jpg"))
        print(f"{split}: {len(data)} images")


if __name__ == "__main__":
    main()
