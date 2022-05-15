import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True, num_classes=100, normalize=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
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


if __name__ == "__main__":
    print(ResNet18(False))
