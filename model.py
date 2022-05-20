from typing import Tuple, Any, Union

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

model_urls = dict(
    vgg11="https://github.com/wonbeomjang/parameters/releases/download/parameter/vgg11_cifar100_teacher.pth",
    resnet18="https://github.com/wonbeomjang/parameters/releases/download/parameter/resnet18_cifar100_teacher.pth",
    resnet50="https://github.com/wonbeomjang/parameters/releases/download/parameter/resnet50_cifar100_teacher.pth",
)


class VGG11(nn.Module):
    output_size = 512 * 7 * 7

    def __init__(self, pretrained=True, num_classes=100, normalize=True, teacher=False):
        super(VGG11, self).__init__()
        backbone = torchvision.models.vgg11(pretrained=pretrained)
        self.normalize = normalize

        self.layer1 = backbone.features[:6]
        self.layer2 = backbone.features[6:11]
        self.layer3 = backbone.features[11:15]
        self.layer4 = backbone.features[15:20]
        self.avgpool = backbone.avgpool

        self.linear = nn.Linear(self.output_size, num_classes)

        if pretrained and teacher:
            self.load_state_dict(load_url(model_urls["vgg11"]))

    def forward(self, x: torch.Tensor, get_ha=False):
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        pool = torch.flatten(pool, 1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, b4, pool, embedding

        return embedding

    def __str__(self):
        return "vgg11"


class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True, num_classes=100, normalize=True, teacher=False):
        super(ResNet18, self).__init__()
        backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.normalize = normalize
        self.linear = nn.Linear(self.output_size, num_classes)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(backbone, module_name))

        if pretrained and teacher:
            self.load_state_dict(load_url(model_urls["resnet18"]))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, b4, pool, embedding

        return embedding

    def __str__(self):
        return "resnet18"


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True, num_classes=100, normalize=True, teacher=False):
        super(ResNet50, self).__init__()
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.normalize = normalize
        self.linear = nn.Linear(self.output_size, num_classes)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(backbone, module_name))

        if pretrained and teacher:
            self.load_state_dict(load_url(model_urls["resnet50"]))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, b4, pool, embedding

        return embedding

    def __str__(self):
        return "resnet50"


if __name__ == "__main__":
    model = VGG11(teacher=True)