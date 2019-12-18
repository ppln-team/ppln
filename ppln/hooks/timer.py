import time

from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class IterTimerHook(BaseHook):
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
