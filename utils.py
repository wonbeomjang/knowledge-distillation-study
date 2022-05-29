import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

def show_attention():
    def make_attention(x: torch.Tensor):
        return F.normalize(x.pow(2).mean(1))

    from model.backbone import VGG11, ResNet50
    from torchvision import transforms
    from torchvision.datasets import CIFAR100
    import matplotlib.pyplot as plt

    class UnNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_train = CIFAR100("../dataset", train=True, transform=train_transform, download=True)

    student = VGG11(teacher=True)
    teacher = ResNet50(teacher=True)

    data: torch.Tensor = dataset_train[0][0]

    student_pred = student(data.unsqueeze(dim=0), True)[:-2]
    teacher_pred = teacher(data.unsqueeze(dim=0), True)[:-2]

    data = unormalize(data)
    adj_factor = 0.9

    for s, t in zip(student_pred, teacher_pred):
        s = make_attention(s).squeeze().repeat(3, 1, 1)
        t = make_attention(t).squeeze().repeat(3, 1, 1)

        s = F.adaptive_avg_pool2d(s, 224)
        t = F.adaptive_avg_pool2d(t, 224)

        s = s * adj_factor + 1 - adj_factor
        t = t * adj_factor + 1 - adj_factor

        s = s * data
        t = t * data

        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(data.permute(1, 2, 0).detach().numpy())

        plt.subplot(1, 3, 2)
        plt.title("student")
        plt.imshow(s.permute(1, 2, 0).detach().numpy())

        plt.subplot(1, 3, 3)
        plt.title("teacher")
        plt.imshow(t.permute(1, 2, 0).detach().numpy())
        plt.show()


def get_attention(x: torch.Tensor):
    return F.normalize(x.pow(2).mean(1))


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
