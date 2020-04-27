import torch.nn as nn
import networks.custom_resnet as custom_resnet
from torchvision import models

from networks.karnet import KarNet
from networks.hidden import Hidden
from networks.testnet import TestNet


classes = 10


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
    if "karnet" == custom_model:
        return KarNet(15)
    if "short_resnet" == custom_model:
        return custom_resnet.short_resnet()
    if "short_many_planes_resnet" == custom_model:
        return custom_resnet.short_many_planes_resnet()
    if "short_many_planes_layers_resnet" == custom_model:
        return custom_resnet.short_many_planes_many_layers_resnet()
    if "long_resnet" == custom_model:
        return custom_resnet.long_resnet()


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
