import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class VGG11(nn.Module):
    output_size = 512 * 7 * 7

    def __init__(self, pretrained=True, num_classes=100, normalize=True):
        super(VGG11, self).__init__()
        pretrained = torchvision.models.vgg11(pretrained=pretrained)
        self.normalize = normalize

        for module_name in ['features', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.linear = nn.Linear(self.output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True, num_classes=100, normalize=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        self.normalize = normalize

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.linear = nn.Linear(self.output_size, num_classes)

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

    def __init__(self, pretrained=True, num_classes=100, normalize=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)
        self.normalize = normalize
        self.linear = nn.Linear(self.output_size, num_classes)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))

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
    print(VGG11(False))