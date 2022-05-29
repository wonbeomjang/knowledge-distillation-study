import os
import math
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))


def get_sample_features(train_loader: DataLoader, student: nn.Module, device: torch.device) -> list[torch.Tensor]:
    features = torch.ones([1])
    for image, target in train_loader:
        image: torch.Tensor = image.to(device)
        features = student(image, True)[:-2]
        break

    return features


def attempt_make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if math.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
