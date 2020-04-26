import torch
import torch.nn as nn
import copy


def _short_forward_impl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def get_short_resnet(resnet, num_classes = 1000):
    short_resnet = copy.deepcopy(resnet)
    short_resnet._forward_impl = _short_forward_impl
    short_resnet.fc = nn.Linear(128, num_classes)
    return short_resnet
