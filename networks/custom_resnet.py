import torch
import torch.nn as nn
import torchvision.models.resnet as resnetm


class CustomResNet(nn.Module):
    """
    ResNet customisation basing on PyToch ResNet implementation.
    See ref: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block=resnetm.BasicBlock, layers=(2, 2, 2, 2), planes=(64, 128, 256, 512), num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CustomResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if layers is None:
            raise ValueError("layers cannot be None")
        if planes is None:
            raise ValueError("planes cannot be None")
        if len(planes) != len(layers):
            raise ValueError("layers and planes should be equal length"
                             "got {} layers and {} planes".format(len(layers), len(planes)))
        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False] * (len(layers) - 1)
        if len(replace_stride_with_dilation) != len(layers) - 1:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a {}-element tuple, got {}".format(len(layers) - 1, replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer = self._make_layer(block, self.inplanes, layers[0])
        
        self.layers_count = len(layers)
        for i, (plane, layer, dilate) in enumerate(zip(planes[1:], layers[1:], replace_stride_with_dilation)):
            self.__setattr__(f'layer{i}', self._make_layer(block, plane, layer, stride=2,
                    dilate=dilate))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnetm.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnetm.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnetm.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer(x)
        for i in range(0, self.layers_count - 1):
            x = self.__getattr__(f'layer{i}')(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def short_resnet():
    return CustomResNet(layers=(2, 2), planes=(256, 512))


def short_many_planes_resnet():
    return CustomResNet(layers=(2, 2), planes=(1024, 1024))


def short_many_planes_many_layers_resnet():
    return CustomResNet(layers=(10, 10), planes=(1024, 1024))


def long_resnet():
    return CustomResNet(layers=(2, 2, 2, 2, 2, 2), planes=(64, 128, 256, 512, 1024, 2048))
