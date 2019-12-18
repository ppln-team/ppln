from torch.nn.utils import clip_grad

from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class OptimizerHook(BaseHook):
    def __init__(self, name='base', grad_clip=None):
        self.name = name
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizers[self.name].zero_grad()
        runner.outputs[f'{self.name}_loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.optimizers[self.name].parameters())
        runner.optimizers[self.name].step()
