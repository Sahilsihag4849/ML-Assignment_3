import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split
from config import SEED, BATCH_SIZE, NUM_WORKERS

SPLIT_PATH = "splits/split.npz"

def _load_cifar(train):
    ds = datasets.CIFAR10("data", train=train, download=True)
    X = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0  # N,3,32,32
    y = torch.tensor(ds.targets)
    return X, y

def get_split():
    """Stratified 45k/5k split of the 50k training images, saved once."""
    if os.path.exists(SPLIT_PATH):
        d = np.load(SPLIT_PATH)
        return d["train"], d["val"]
    _, y = _load_cifar(True)
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=5000, stratify=y.numpy(),
                              random_state=SEED)
    os.makedirs("splits", exist_ok=True)
    np.savez(SPLIT_PATH, train=tr, val=va)
    return tr, va

def load_train_val():
    X, y = _load_cifar(True)
    tr, va = get_split()
    return X[tr], y[tr], X[va], y[va]

def load_test():
    """Use ONLY from final evaluation scripts."""
    return _load_cifar(False)

def norm_stats(X_train):
    """Mean/std from the 45k training split only."""
    return X_train.mean(dim=(0, 2, 3)), X_train.std(dim=(0, 2, 3))

class Normalize:
    def __init__(self, mean, std):
        self.m = mean.view(3, 1, 1); self.s = std.view(3, 1, 1)
    def __call__(self, x):
        return (x - self.m) / self.s

class PixelPermute:
    """Fixed spatial shuffle: same permutation for every image, all channels."""
    def __init__(self, seed=SEED):
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(32 * 32, generator=g)
    def __call__(self, x):
        return x.reshape(3, -1)[:, self.perm].reshape(3, 32, 32)

class Compose:
    def __init__(self, fns): self.fns = fns
    def __call__(self, x):
        for f in self.fns: x = f(x)
        return x

class TensorDS(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.t = X, y, transform
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.t is not None: x = self.t(x)
        return x, self.y[i]

def make_loader(X, y, transform=None, shuffle=False, batch_size=BATCH_SIZE):
    return DataLoader(TensorDS(X, y, transform), batch_size=batch_size,
                      shuffle=shuffle, num_workers=NUM_WORKERS)