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
        global classes
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


class Hidden(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super(Hidden, self).__init__()
        if len(hidden_sizes) == 0:
            raise ValueError('List of hidden layers sizes cannot be empty')
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fch = [
            nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(1, len(hidden_sizes))
        ]
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        for fc in self.fch:
            x = fc(x)
        x = self.fc2(x)
        return x


def freeze_parameters(model):
    if model:
        for param in model.parameters():
            param.requires_grad = False


def get_custom_model(custom_model, input_size):
    global classes
    if "linear" == custom_model:
        return nn.Linear(input_size, classes)
    if "testnet" == custom_model:
        return TestNet()
    if "hidden100" == custom_model:
        return Hidden(input_size, [100], classes)


def build(transfer_model_name, custom_model_name, freeze_transfer):
    model = None
    set_last_layer = None
    num_ftrs = 0
    input_size = 32

    if transfer_model_name == "resnet":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        input_size = 224
        set_last_layer = lambda model, cm: exec("model.fc = cm")

    elif transfer_model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        input_size = 224
        set_last_layer = lambda model, cm: exec("model.classifier[6] = cm")

    elif transfer_model_name == "vgg":
        model = models.vgg19_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        input_size = 224
        set_last_layer = lambda model, cm: exec("model.classifier[6] = cm")

    elif transfer_model_name == "densenet":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        input_size = 224
        set_last_layer = lambda model, cm: exec("model.classifier = cm")

    if freeze_transfer:
        freeze_parameters(model)

    custom_model = get_custom_model(custom_model_name, num_ftrs)

    if model:
        set_last_layer(model, custom_model)
    else:
        model = custom_model

    return model, input_size
