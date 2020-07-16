import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class LRSchedulerHook(BaseHook):
    def __init__(self, monitor_metric="loss", by_epoch=True):
        self.monitor_metric = monitor_metric
        self.by_epoch = by_epoch

    def after_val_epoch(self, runner):
        if self.by_epoch:
            self.step(runner)

    def after_train_iter(self, runner):
        if not self.by_epoch:
            self.step(runner)

    def step(self, runner):
        with warnings.catch_warnings():
            # https://discuss.pytorch.org/t/cyclic-learning-rate-how-to-use/53796/2
            warnings.filterwarnings("ignore", category=UserWarning)
            if isinstance(runner.scheduler, ReduceLROnPlateau):
                runner.scheduler.step(runner.epoch_outputs[self.monitor_metric])
            else:
                runner.scheduler.step()
