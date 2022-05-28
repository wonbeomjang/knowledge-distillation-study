import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def forward(self, image, target, student, teacher):
        pred = student(image)
        return F.cross_entropy(pred, target), pred