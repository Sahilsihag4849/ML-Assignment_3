import sys; sys.path.insert(0, "src")
from config import SEED
from utils import set_seed, count_params
from data import load_train_val, norm_stats, Normalize, make_loader
from models import build_mlp
from train import fit

set_seed(SEED)
Xtr, ytr, Xva, yva = load_train_val()
norm = Normalize(*norm_stats(Xtr))
tr = make_loader(Xtr, ytr, norm, shuffle=True)
va = make_loader(Xva, yva, norm)
m = build_mlp()
print("params:", count_params(m))
fit(m, tr, va, epochs=1, name="smoke")