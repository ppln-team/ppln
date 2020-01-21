from torch.nn.utils import clip_grad

from .base import BaseClosureHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class OptimizerHook(BaseClosureHook):
    @property
    def priority(self):
        return Priority.HIGH

    def __init__(self, name='base', **kwargs):
        super().__init__(clip_grad.clip_grad_norm_, **kwargs)
        self.name = name
        self.is_clip = len(kwargs) > 0

    def after_train_iter(self, runner):
        runner.optimizers[self.name].zero_grad()
        runner.outputs[f'{self.name}_loss'].backward()
        if self.is_clip:
            for param_group in runner.optimizers[self.name].param_groups:
                self.func(param_group['params'])
        runner.optimizers[self.name].step()
