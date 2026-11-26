import torch

ROLL_NUMBER = "M26MAC008"
SEED = int("".join(c for c in ROLL_NUMBER if c.isdigit()))   # all numeric chars, in order
DEVICE = "cpu"