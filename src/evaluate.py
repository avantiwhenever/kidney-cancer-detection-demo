"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python src/evaluate.py --checkpoint checkpoints/resnet18_kidney.pt --data-dir data/raw
"""
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from dataset import load_split
from model import build_model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = load_split(args.data_dir, "test", train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    tumor_idx = classes.index("Tumor")
    probs = torch.cat(all_probs).numpy()
    tumor_binary = [1 if l == tumor_idx else 0 for l in all_labels]
    auc = roc_auc_score(tumor_binary, probs[:, tumor_idx])
    print(f"Tumor-vs-rest ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--batch-size", type=int, default=32)
    evaluate(parser.parse_args())
