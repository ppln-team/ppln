from ..factory import make_apex_ddp, make_pytorch_ddp
from .base import BaseClosureHook, BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class DistSamplerSeedHook(BaseHook):
    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)


@HOOKS.register_module
class BaseDDPHook(BaseHook):
    @property
    def priority(self):
        return Priority.VERY_HIGH


@HOOKS.register_module
class PytorchDDPHook(BaseClosureHook, BaseDDPHook):
    def __init__(self, **kwargs):
        super().__init__(make_pytorch_ddp, **kwargs)

    def before_run(self, runner):
        runner.model = self.func(runner.model)


@HOOKS.register_module
class ApexDDPHook(BaseClosureHook, BaseDDPHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex_ddp, **kwargs)

    def before_run(self, runner):
        runner.model = self.func(runner.model)
