import torch.nn as nn

def build_mlp(hidden=(512, 256), num_classes=10, p_drop=0.0):
    layers, d = [nn.Flatten()], 3 * 32 * 32
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if p_drop > 0: layers.append(nn.Dropout(p_drop))
        d = h
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)

def build_cnn(channels=(32, 64, 128), num_classes=10):
    layers, c_in = [], 3
    for c in channels:
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c),
                   nn.ReLU(), nn.MaxPool2d(2)]
        c_in = c
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, num_classes)]
    return nn.Sequential(*layers)