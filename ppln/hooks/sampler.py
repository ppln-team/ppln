from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class DistSamplerSeedHook(BaseHook):
    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)
