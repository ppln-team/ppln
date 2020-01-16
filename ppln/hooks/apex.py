import warnings

from ..factory import make_apex
from .base import BaseClosureHook
from .optimizer import OptimizerHook
from .priority import Priority
from .registry import HOOKS

try:
    from apex import amp
except ImportError as e:
    warnings.warn(
        f"Error \"{e}\" during importing apex library. To use mixed precison"
        ' you should install it from https://github.com/NVIDIA/apex'
    )


@HOOKS.register_module
class ApexOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizers[self.name].zero_grad()

        with amp.scale_loss(runner.outputs[f'{self.name}_loss'], runner.optimizers[self.name]) as scaled_loss:
            scaled_loss.backward()

        if self.is_clip:
            self.func(amp.master_params(runner.optimizers[self.name]))
        runner.optimizers[self.name].step()


@HOOKS.register_module
class ApexInitializeHook(BaseClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex, **kwargs)

    @property
    def priority(self):
        return Priority.HIGHEST

    def before_run(self, runner):
        runner.model, optimizers = self.func(runner.model, list(runner.optimizers.values()))
        for name, optimizer in zip(runner.optimizers, optimizers):
            runner.optimizers[name] = optimizer
