from typing import List, NoReturn, Optional

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def freeze_modules(
    model: nn.Module,
    root_modules: List[str],
    requires_grad: bool = False,
    train: bool = False,
    parent_module: Optional[str] = None,
) -> NoReturn:
    """
    Select root modules from list(model.named_modules()) that will be frozen recursively.
    Make sure all top layers are frozen without gaps.
    The function allows you to freeze and unfreeze selected modules for any model
    (the train mode setting works similarly)
    """
    for name, child in model.named_children():
        if name == "module":
            freeze_modules(child, root_modules, requires_grad, train)
        current_module = name if parent_module is None else f"{parent_module}.{name}"

        if current_module in root_modules:
            child.train(train)
            for param in child.parameters():
                param.requires_grad = requires_grad
        else:
            freeze_modules(child, root_modules, requires_grad, train, current_module)


def lock_norm_modules(model: nn.Module, train: bool = False, requires_grad: Optional[bool] = None) -> NoReturn:
    """
    Changes the train mode for all normalizing layers of the model.
    It should be used after freezing the network.
    If the requires_grad is not specified, then leaves it unchanged.
    """
    norm_modules = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm, nn.LocalResponseNorm)

    for module in model.modules():
        if isinstance(module, norm_modules):
            module.train(train)
            if requires_grad is not None:
                for param in module.parameters():
                    param.requires_grad = requires_grad
