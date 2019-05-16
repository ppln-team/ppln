from torch import nn


class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cSE = nn.Sequential(
            nn.Linear(planes, planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        cse = self.avgpool(x).view(b, c)
        cse = self.cSE(cse).view(b, c, 1, 1)
        cse = x * cse

        return cse
