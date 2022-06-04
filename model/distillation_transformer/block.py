import torch
from torch import nn as nn
from torch.nn import functional as F


class InReshapeBlock(nn.Module):
    def __init__(self, size):
        super(InReshapeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        feature = [self.avg_pool(t) for t in x]
        feature = torch.stack(feature, 1)
        return feature


class OutReshapeBlock(nn.Module):
    def __init__(self):
        super(OutReshapeBlock, self).__init__()

    def forward(self, x, trg):
        features = []
        for i in range(x.size(1)):
            features += [F.adaptive_avg_pool2d(x[:, i], trg[i].size(1))]

        return features
