from data import load_test, make_loader
from train import evaluate

def evaluate_on_test(model, norm, transform_extra=None):
    X, y = load_test()
    t = norm if transform_extra is None else (lambda x: norm(transform_extra(x)))
    return evaluate(model, make_loader(X, y, t))