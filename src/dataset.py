"""PyTorch Dataset for the kidney CT classification data.

Expects images laid out as data/raw/<split>/<class_name>/*.jpg, which is what
data/download_data.py produces from the mhmad240/kidney-ct-classification
Hugging Face dataset.
"""
import torchvision.datasets as datasets
import torchvision.transforms as T

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE = 224


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_split(root, split, train=False):
    return datasets.ImageFolder(f"{root}/{split}", transform=get_transforms(train))
