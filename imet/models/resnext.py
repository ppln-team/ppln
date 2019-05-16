import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .resnet import Bottleneck as ResNetBottleneck, ResNet


__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext101_64', 'resnext152']


model_urls = {
    'resnext50': 'https://easygold.ai/pretrained/resnext50-316de15a.pth',
    'resnext101': 'https://easygold.ai/pretrained/resnext101-a04abaaf.pth'
}


class Bottleneck(ResNetBottleneck):
    """
    [SC][SE-]ResNeXt bottleneck type C
    """
    def __init__(self, inplanes, planes, baseWidth, cardinality,
                 stride=1, downsample=None, reduction=16, scse=None):
        super(Bottleneck, self).__init__(inplanes, planes, stride, downsample,
                                         reduction, scse)

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)


class ResNeXt(ResNet):

    def __init__(self, block, baseWidth, cardinality, layers, num_classes=1000):
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        super(ResNeXt, self).__init__(block, layers, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth,
                            self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)


def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext50']))
    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 (32x4d) model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101']))
    return model


def resnext101_64(**kwargs):
    """Constructs a ResNeXt-101 (64x4d) model."""
    model = ResNeXt(Bottleneck, 4, 64, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 (32x4d) model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 8, 36, 3], **kwargs)
    return model
