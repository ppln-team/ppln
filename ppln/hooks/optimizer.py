from torch.nn.utils.clip_grad import clip_grad_norm_

from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class OptimizerHook(BaseHook):
    @property
    def priority(self):
        return Priority.HIGH

    def __init__(self, max_norm=None, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.batch_outputs["loss"].backward()
        if self.max_norm is not None:
            for param_group in runner.optimizer.param_groups:
                clip_grad_norm_(param_group["params"], max_norm=self.max_norm, norm_type=self.norm_type)
        runner.optimizer.step()
