import time, torch, torch.nn as nn
from config import DEVICE
from utils import save_json, count_params

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    tot_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        tot_loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / n, correct / n

def fit(model, train_loader, val_loader, epochs, lr=1e-3, weight_decay=0.0,
        log_path=None, name="run"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        # measure train metrics in eval mode (clean, comparable to val)
        tl, ta = evaluate(model, train_loader)
        vl, va = evaluate(model, val_loader)
        for k, v in zip(hist, (tl, ta, vl, va)): hist[k].append(v)
        print(f"[{name}] ep {ep+1}/{epochs} train {ta:.3f} val {va:.3f} "
              f"({time.time()-t0:.0f}s)")
    if log_path:
        save_json({"name": name, "params": count_params(model), "lr": lr,
                   "weight_decay": weight_decay, "epochs": epochs,
                   "history": hist}, log_path)
    return hist