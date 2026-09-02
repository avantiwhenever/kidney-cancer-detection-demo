"""Fetch a balanced subset of each split directly via the Hugging Face
datasets-server API (stdlib only, no `datasets`/`huggingface_hub` install
needed) and lay it out as data/raw/<split>/<class>/*.jpg for
torchvision.ImageFolder. Useful for a fast smoke-training run or CI without
pulling the full ~12k-image dataset.

For the full dataset, use download_data.py instead.

Usage:
    python data/fetch_subset.py --per-class 250 60 60
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SPLITS = ["train", "validation", "test"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=mhmad240/kidney-ct-classification&config=default&split={split}"
    "&offset={offset}&length=100"
)


def _retrieve_with_retry(url, dest, attempts=4):
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_split(split, per_class, max_offset=10000):
    out_root = os.path.join(OUT_DIR, split)
    # Track by filename, not a plain counter -- a resumed scan revisits
    # already-downloaded rows (offset restarts at 0), and a counter would
    # double-count those instead of just confirming they're already done.
    seen = {}
    for c in CLASSES:
        class_dir = os.path.join(out_root, c)
        seen[c] = set(os.listdir(class_dir)) if os.path.isdir(class_dir) else set()

    offset = 0
    while offset < max_offset and not all(len(v) >= per_class for v in seen.values()):
        with urllib.request.urlopen(API.format(split=split, offset=offset)) as resp:
            payload = json.load(resp)
        if not payload["rows"]:
            break
        for row in payload["rows"]:
            label = CLASSES[row["row"]["label"]]
            fname = f"{row['row_idx']}.jpg"
            if len(seen[label]) >= per_class or fname in seen[label]:
                continue
            out_dir = os.path.join(out_root, label)
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, fname)
            if not os.path.exists(dest):
                _retrieve_with_retry(row["row"]["image"]["src"], dest)
            seen[label].add(fname)
        offset += 100
    print(f"{split}: {{{', '.join(f'{c!r}: {len(seen[c])}' for c in CLASSES)}}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class", type=int, nargs=3, default=[250, 60, 60],
        metavar=("TRAIN", "VAL", "TEST"),
        help="images per class for train/validation/test",
    )
    args = parser.parse_args()
    for split, n in zip(SPLITS, args.per_class):
        fetch_split(split, n)


if __name__ == "__main__":
    main()
