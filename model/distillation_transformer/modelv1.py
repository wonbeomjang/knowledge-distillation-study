import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from model.distillation_transformer.block import InReshapeBlock, OutReshapeBlock

ckpt_url = "https://github.com/wonbeomjang/parameters/releases/download/parameter/transfrom.pth"


class ConvV1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=torch.device("cpu")):
        super(ConvV1, self).__init__()
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

    def forward(self, x):
        x = self.residual(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, -1)

        query = self.query(y)
        key = self.key(y)
        value = self.value(y)

        # (b c 1) x (b 1 1) -> (b c c)
        attention = torch.bmm(query.view(b, c, 1), key.view(b, 1, c))
        attention = F.softmax(attention, dim=-1)
        attention = attention * self.mask

        # (b c c) x (b c 1 ) -> (b c 1)
        y = torch.bmm(attention, value.view(b, c, 1))

        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=torch.device("cpu")):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvV1(in_channels, out_channels, 3, 1, 1, device),
            ConvV1(out_channels, out_channels, 3, 1, 1, device)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=torch.device("cpu")):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.block = nn.Sequential(
            ConvV1(in_channels, out_channels, 3, 1, 1, device),
            ConvV1(out_channels, out_channels, 3, 1, 1, device)
        )

    def forward(self, x, feature):
        x = self.upsample(x)
        x = torch.cat([x, feature], dim=1)
        x = self.block(x)
        return x


class Transform(nn.Module):
    def __init__(self, in_channels, out_channels, size=56, device=torch.device("cpu"), pretrained=False):
        super(Transform, self).__init__()

        self.in_reshape = InReshapeBlock(size)
        self.out_reshape = OutReshapeBlock()

        self.adj = nn.Sequential(
            ConvV1(in_channels, 64, 3, 1, 1, device),
            ConvV1(64, 64, 3, 1, 1, device),
        )

        self.encoder1 = EncoderBlock(64, 128, device)
        self.encoder2 = EncoderBlock(128, 256, device)
        self.encoder3 = EncoderBlock(256, 512, device)

        self.decoder1 = DecoderBlock(512, 256, device)
        self.decoder2 = DecoderBlock(256, 128, device)
        self.decoder3 = DecoderBlock(128, 64, device)

        self.out = nn.Conv2d(64, out_channels, 1)

        if pretrained:
            self.load_state_dict(load_url(ckpt_url))

    def forward(self, src, trg):
        x = self.in_reshape(src)

        x1 = self.adj(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.out(x)

        x = self.out_reshape(x, trg)
        return x
