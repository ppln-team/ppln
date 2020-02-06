import numpy as np

from ..utils.misc import master_only
from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class EarlyStoppingHook(BaseHook):
    def __init__(self, metric_name='base_loss', patience=0, verbose=0, mode='min'):
        super().__init__()

        self.metric_name = metric_name
        self.patience = patience
        self.verbose = verbose

        if mode == 'min':
            self._is_update = np.less
        elif mode == 'max':
            self._is_update = np.greater

        self._wait = 0
        self._stopped_epoch = 0
        self._best_metric = np.infty if self._is_update == np.less else -np.infty

    @property
    def priority(self):
        return Priority.LOWEST

    def after_val_epoch(self, runner):
        metric = runner.log_buffer.output[self.metric_name]

        if self._is_update(metric, self._best_metric):
            self._best_metric = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stopped_epoch = runner.epoch
                runner.stop_training = True

    @master_only
    def after_run(self, runner):
        if self._stopped_epoch > 0 and self.verbose > 0:
            runner.logger.info(f'Epoch {self._stopped_epoch + 1}: early stopping')
