from torch import nn

from .se import SEBlock


class SCSEBlock(SEBlock):
    def __init__(self, planes, reduction=16):
        super(SCSEBlock, self).__init__(planes, reduction)

        self.sSE = nn.Sequential(
            nn.Conv2d(planes, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        cse = super(SCSEBlock, self).forward(x)

        sse = self.sSE(x)
        sse = x * sse

        return cse + sse
