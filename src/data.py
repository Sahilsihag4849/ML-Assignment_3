import os, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from config import SEED

def get_split():
    path = "splits/split.npz"
    if os.path.exists(path):
        d = np.load(path); return d["train"], d["val"]
    full = datasets.CIFAR10("data", train=True, download=True)
    idx = np.arange(len(full))
    tr, va = train_test_split(idx, test_size=5000, stratify=full.targets,
                              random_state=SEED)
    os.makedirs("splits", exist_ok=True)
    np.savez(path, train=tr, val=va)
    return tr, va