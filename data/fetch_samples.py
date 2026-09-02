"""Pull a small real sample (a handful of images per class) directly from the
Hugging Face datasets-server API, no extra dependencies required. Used to
seed data/sample/ so the repo has real images to smoke-test the pipeline
against without downloading the full ~8.7k-image dataset. For the full
dataset, use download_data.py instead.

Usage:
    python data/fetch_samples.py
"""
import json
import os
import urllib.request

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
PER_CLASS = 6
OUT_DIR = os.path.join(os.path.dirname(__file__), "sample")
API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=mhmad240/kidney-ct-classification&config=default&split=train"
    "&offset={offset}&length=100"
)


def main():
    counts = {c: 0 for c in CLASSES}
    offset = 0
    while offset < 3000 and not all(v >= PER_CLASS for v in counts.values()):
        with urllib.request.urlopen(API.format(offset=offset)) as resp:
            payload = json.load(resp)
        for row in payload["rows"]:
            label = CLASSES[row["row"]["label"]]
            if counts[label] >= PER_CLASS:
                continue
            url = row["row"]["image"]["src"]
            out_dir = os.path.join(OUT_DIR, label)
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{row['row_idx']}.jpg")
            urllib.request.urlretrieve(url, dest)
            counts[label] += 1
        offset += 100
    print("Downloaded:", counts)


if __name__ == "__main__":
    main()
