import torch.nn as nn

from ppln.utils.freeze import freeze_modules, lock_norm_modules


class LinearBNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.ReLU())
        self.bn = nn.BatchNorm1d(2)


def test_freeze_modules():
    model = LinearBNModule()
    freeze_modules(model, root_modules=["layer1"], train=False)

    assert not model.layer1.training
    assert model.bn.training
    for p in model.layer1.parameters():
        assert not p.requires_grad
    for p in model.bn.parameters():
        assert p.requires_grad


def test_lock_norm_modules():
    model = LinearBNModule()
    lock_norm_modules(model, requires_grad=False, train=False)

    assert model.layer1.training
    assert not model.bn.training
    for p in model.layer1.parameters():
        assert p.requires_grad
    for p in model.bn.parameters():
        assert not p.requires_grad
