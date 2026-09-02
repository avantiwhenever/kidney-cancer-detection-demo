"""Turn a flat folder of your own images (organized one subfolder per class)
into the train/validation/test layout that src/train.py and src/evaluate.py
expect. Works for any image classification task, not just kidney CT --
whatever class folders you provide become the model's classes.

Input:
    my_data/
        ClassA/*.jpg
        ClassB/*.jpg
        ...

Output (default):
    data/raw/train/ClassA/*.jpg ...
    data/raw/validation/ClassA/*.jpg ...
    data/raw/test/ClassA/*.jpg ...

Usage:
    python data/prepare_own_data.py --input-dir /path/to/my_data
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_class(files, val_frac, test_frac, seed):
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    return {
        "test": files[:n_test],
        "validation": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="folder with one subfolder per class")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy", action="store_true",
        help="copy files instead of symlinking (use if input-dir may move/be deleted)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    class_dirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {input_dir}")

    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if not files:
            print(f"skipping {class_dir.name}: no images found")
            continue
        splits = split_class(files, args.val_frac, args.test_frac, args.seed)
        for split, split_files in splits.items():
            dest_dir = output_dir / split / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                dest = dest_dir / f.name
                if dest.exists():
                    continue
                if args.copy:
                    shutil.copy2(f, dest)
                else:
                    dest.symlink_to(f.resolve())
        print(f"{class_dir.name}: train={len(splits['train'])} "
              f"validation={len(splits['validation'])} test={len(splits['test'])}")

    print(f"\nDone. Data ready at {output_dir}/{{train,validation,test}}/<class>/")


if __name__ == "__main__":
    main()
