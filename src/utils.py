import random, numpy as np, torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)