import warnings

import torch.utils.model_zoo as model_zoo

from .resnext import Bottleneck, ResNeXt


__all__ = ['se_resnext50', 'se_resnext101',
           'se_resnext101_64', 'se_resnext152']


model_urls = {
    'se_resnext50': 'https://easygold.ai/pretrained/se_resnext50-5cc09937.pth',
    'se_resnext101': 'https://easygold.ai/pretrained/se_resnext101-bee3dc76.pth',
    'se_resnext152': 'https://easygold.ai/pretrained/se_resnext152-57569907.pth'
}


class SEBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        kwargs.pop('scse', None)
        super().__init__(*args, **kwargs, scse='se')


class SEBottleneckR2(SEBottleneck):
    def __init__(self, *args, **kwargs):
        kwargs.pop('reduction', None)
        super().__init__(*args, **kwargs, reduction=2)


def se_resnext50(pretrained=False, **kwargs):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnext50']))
    return model


def se_resnext101(pretrained=False, **kwargs):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnext101']))
    return model


def se_resnext101_64(**kwargs):
    """Constructs a SE-ResNeXt-101 (64x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 64, [3, 4, 23, 3], **kwargs)
    return model


def se_resnext152(pretrained=False, **kwargs):
    """Constructs a SE-ResNeXt-152 (32x4d) model."""
    if pretrained:
        warnings.warn('Using "se_resnext152 n01z3 edition" with reduction=2')
        model = ResNeXt(SEBottleneckR2, 4, 32, [3, 8, 36, 3], **kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnext152']))
    else:
        model = ResNeXt(SEBottleneck, 4, 32, [3, 8, 36, 3], **kwargs)
    return model
