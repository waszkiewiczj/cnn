import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

classes = 10


class TestNet(nn.Module):
    """
    Network taken from PyTorch repository tutorials.
    See ref: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
    """

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestGpuNet(nn.Module):
    """
    Network taken from PyTorch repository tutorials upgraded to be efficciently computed on GPU
    (first convolution layer filter size to 500 features).
    See ref: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
    """

    def __init__(self):
        super(TestGpuNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 500, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(500, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def ResNet():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, classes)
    return model_ft
