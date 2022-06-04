import torch
import torch.nn as nn

from .block import *


class ConvV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=torch.device("cpu")):
        super(ConvV2, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.query = nn.Linear(out_channels, out_channels)
        self.key = nn.Linear(out_channels, out_channels)
        self.value = nn.Linear(out_channels, out_channels)
        self.mask = torch.zeros((out_channels, out_channels), device=device, requires_grad=False)
        for i in range(out_channels):
            self.mask[i][:i + 1] = 1

    def forward(self, query, key, value):
        x = self.residual(query)
        b, c, _, _ = x.size()

        query = self.avg_pool(query).view(b, -1)
        key = self.avg_pool(key).view(b, -1)
        value = self.avg_pool(value).view(b, -1)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (b c 1) x (b 1 1) -> (b c c)
        attention = torch.bmm(query.view(b, c, 1), key.view(b, 1, c))
        attention = F.softmax(attention, dim=-1)
        attention = attention * self.mask

        # (b c c) x (b c 1 ) -> (b c 1)
        y = torch.bmm(attention, value.view(b, c, 1))

        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class EncoderV2(nn.Module):
    def __init__(self, channels, device):
        super(EncoderV2, self).__init__()
        self.layer1 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer2 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer3 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer4 = ConvV2(channels, channels, 3, 1, 1, device)

    def forward(self, x):
        x = self.layer1(x, x, x)
        x = self.layer2(x, x, x)
        x = self.layer3(x, x, x)
        x = self.layer4(x, x, x)
        return x


class DecoderV2(nn.Module):
    def __init__(self, channels, device):
        super(DecoderV2, self).__init__()
        self.layer1 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer2 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer3 = ConvV2(channels, channels, 3, 1, 1, device)
        self.layer4 = ConvV2(channels, channels, 3, 1, 1, device)

        self.adj = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, y, x):
        y = self.layer1(y, x, x)
        y = self.layer2(y, x, x)
        y = self.layer3(y, x, x)
        y = self.layer4(y, x, x)
        y = self.adj(y)
        return y


class TransformV2(nn.Module):
    def __init__(self, in_channels, out_channels, size=56, device=torch.device("cpu"), pretrained=False):
        super(TransformV2, self).__init__()
        assert in_channels == out_channels

        self.in_reshape = InReshapeBlock(size)
        self.out_reshape = OutReshapeBlock()

        self.encoder = EncoderV2(in_channels, device)
        self.decoder = DecoderV2(in_channels, device)

    def forward(self, src, trg):
        x = self.in_reshape(src)
        y = self.in_reshape(trg)

        x = self.encoder(x)
        x = self.decoder(y, x)

        x = self.out_reshape(x, trg)
        return x
