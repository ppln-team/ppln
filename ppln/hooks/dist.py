import warnings

import torch
from torch.nn import SyncBatchNorm

from ..factory import make_apex_ddp, make_pytorch_ddp, make_pytorch_dp, make_pytorch_bdp
from .base import BaseClosureHook
from .priority import Priority
from .registry import HOOKS

try:
    from apex.parallel import convert_syncbn_model as apex_convert_sync_batchnorm
except ImportError as e:
    warnings.warn(
        f'Error "{e}" during importing apex library. To use mixed precison'
        " you should install it from https://github.com/NVIDIA/apex"
    )


@HOOKS.register_module
class ModelClosureHook(BaseClosureHook):
    @property
    def priority(self):
        return Priority.HIGH

    def before_run(self, runner):
        model_device = next(runner.model.parameters()).device
        if model_device != torch.device("cpu"):
            runner.model = self._func(runner.model)


@HOOKS.register_module
class PytorchDPHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_pytorch_dp, **kwargs)


@HOOKS.register_module
class PytorchBDPHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_pytorch_bdp, **kwargs)


@HOOKS.register_module
class PytorchDDPHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_pytorch_ddp, **kwargs)


@HOOKS.register_module
class ApexDDPHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex_ddp, **kwargs)


@HOOKS.register_module
class ApexSyncBNHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(apex_convert_sync_batchnorm, **kwargs)


@HOOKS.register_module
class PytorchSyncBNHook(ModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(SyncBatchNorm.convert_sync_batchnorm, **kwargs)
