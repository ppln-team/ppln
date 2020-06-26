import warnings

from torch.nn import SyncBatchNorm

from ..factory import make_apex_ddp, make_pytorch_ddp
from .base import BaseClosureHook
from .registry import HOOKS

try:
    from apex.parallel import convert_syncbn_model as apex_convert_sync_batchnorm
except ImportError as e:
    warnings.warn(
        f'Error "{e}" during importing apex library. To use mixed precison'
        " you should install it from https://github.com/NVIDIA/apex"
    )


@HOOKS.register_module
class BaseModelClosureHook(BaseClosureHook):
    def before_run(self, runner):
        runner.model = self.func(runner.model)


@HOOKS.register_module
class PytorchDDPHook(BaseModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_pytorch_ddp, **kwargs)


@HOOKS.register_module
class ApexDDPHook(BaseModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex_ddp, **kwargs)


@HOOKS.register_module
class ApexSyncBNHook(BaseModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(apex_convert_sync_batchnorm, **kwargs)


@HOOKS.register_module
class PytorchSyncBNHook(BaseModelClosureHook):
    def __init__(self, **kwargs):
        super().__init__(SyncBatchNorm.convert_sync_batchnorm, **kwargs)
