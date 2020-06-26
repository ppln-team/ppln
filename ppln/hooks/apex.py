import warnings

from torch.nn.utils.clip_grad import clip_grad_norm_

from ..factory import make_apex
from .base import BaseClosureHook
from .optimizer import OptimizerHook
from .priority import Priority
from .registry import HOOKS

try:
    from apex import amp
except ImportError as e:
    warnings.warn(
        f'Error "{e}" during importing apex library. To use mixed precison'
        " you should install it from https://github.com/NVIDIA/apex"
    )


@HOOKS.register_module
class ApexOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        with amp.scale_loss(runner.batch_processor_output["loss"], runner.optimizer) as scaled_loss:
            scaled_loss.backward()

        if isinstance(self.max_norm, float):
            clip_grad_norm_(amp.master_params(runner.optimizer), max_norm=self.max_norm, norm_type=self.norm_type)
        runner.optimizer.step()


@HOOKS.register_module
class ApexInitializeHook(BaseClosureHook):
    def __init__(self, **kwargs):
        super().__init__(make_apex, **kwargs)

    @property
    def priority(self):
        return Priority.VERY_HIGH

    def before_run(self, runner):
        runner.model, runner.optimizer = self.func(runner.model, runner.optimizer)
