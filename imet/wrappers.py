import torch.nn as nn

from .models import se_resnext


class TwoHeadSEResNeXt(nn.Module):
    def __init__(self, num_tag_classes, num_culture_classes, architecture='se_resnext50', pretrained=True):
        super().__init__()
        self.architecture = architecture
        self.pretrained = pretrained
        self.num_culture_classes = num_culture_classes
        self.num_tag_classes = num_tag_classes
        model = getattr(se_resnext, self.architecture)(pretrained=pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool)
        self.tag_classifier = nn.Linear(2048, num_tag_classes)
        self.culture_classifier = nn.Linear(2048, num_culture_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        tag_x = self.tag_classifier(x)
        culture_x = self.culture_classifier(x)
        return tag_x, culture_x


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, x, tag_labels, culture_labels):
        tag_x, culture_x = self.model(x)
        tag_loss = self.loss(tag_x, tag_labels)
        culture_loss = self.loss(culture_x, culture_labels)
        return tag_loss, culture_loss
