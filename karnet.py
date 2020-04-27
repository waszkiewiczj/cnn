import torch.nn as nn
import math
import numpy as np
import torch
from helpers import get_device


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.vote = VoteBlock(planes)

    def forward(self, x):
        out = x[0]
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        block_vote = self.vote(out)
        votes = x[1]
        if votes!=None:
            votes += block_vote
        else:
            votes = block_vote

        return [out, votes]


class VoteBlock(nn.Module):

    def __init__(self, input_size):
        super(VoteBlock, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(input_size*4*4, 50)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class KarNet(nn.Module):

    def __init__(self, num_blocks):
        super(KarNet, self).__init__()
        self.num_blocks = num_blocks
        size = 64
        self.first = BasicBlock(3, size)
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(size, size))

        self.blocks = nn.Sequential(*layers)
        

    def forward(self, x):
        x, linear = self.first([x, None])
        x, linear = self.blocks([x, linear])

        return linear
