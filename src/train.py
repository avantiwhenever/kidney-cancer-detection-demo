"""Fine-tune a ResNet18 classifier on the kidney CT dataset.

Usage:
    python src/train.py --data-dir data/raw --epochs 10
"""
import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import load_split
from model import build_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_split(args.data_dir, "train", train=True)
    val_ds = load_split(args.data_dir, "validation", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"epoch {epoch + 1}/{args.epochs} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, args.checkpoint)

    print(f"best val_acc={best_acc:.4f}, saved to {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_kidney.pt")
    train(parser.parse_args())
