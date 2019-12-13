import warnings

from ..factory import make_apex
from .base import BaseClosureHook, BaseHook
from .optimizer import OptimizerHook
from .priority import Priority

try:
    from apex import amp
except ImportError as e:
    warnings.warn(
        f"Error \"{e}\" during importing apex library. To use mixed precison"
        ' you should install it from https://github.com/NVIDIA/apex'
    )


class ApexOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
            scaled_loss.backward()

        if self.grad_clip is not None:
            self.clip_grads(amp.master_params(runner.optimizer))
        runner.optimizer.step()


class ApexInitializeHook(BaseClosureHook, BaseHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex, **kwargs)

    @property
    def priority(self):
        return Priority.HIGHEST

    def before_run(self, runner):
        runner.model, runner.optimizer = self.func(runner.model, runner.optimizer)
