from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class LRSchedulerHook(BaseHook):
    def __init__(self, metric_name="loss", by_epoch=True):
        self.metric_name = metric_name
        self.by_epoch = by_epoch

    def after_val_epoch(self, runner):
        if self.by_epoch:
            self.step(runner)

    def after_train_iter(self, runner):
        if not self.by_epoch:
            self.step(runner)

    def step(self, runner):
        if isinstance(runner.scheduler, ReduceLROnPlateau):
            runner.scheduler.step(runner.epoch_outputs[self.metric_name])
        else:
            runner.scheduler.step()
