import numpy as np

from ..utils.dist import master_only
from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class EarlyStoppingHook(BaseHook):
    def __init__(self, monitor_metric="loss", mode="min", patience=0):
        super().__init__()
        self.monitor_metric = monitor_metric
        self.patience = patience

        if mode == "min":
            self._compare_metrics = np.less
            self._best_metric = np.infty
        elif mode == "max":
            self._compare_metrics = np.greater
            self._best_metric = -np.infty

        self._wait = 0

    @property
    def priority(self):
        return Priority.LOWEST

    def after_val_epoch(self, runner):
        metric = runner.epoch_outputs[self.monitor_metric]

        if self._compare_metrics(metric, self._best_metric):
            self._best_metric = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                runner.stop_training = True

    @master_only
    def after_run(self, runner):
        if runner.stop_training:
            runner.logger.info(f"Early stopping")
