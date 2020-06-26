import time

from .base import BaseHook
from .registry import HOOKS


@HOOKS.register_module
class TimerHook(BaseHook):
    def __init__(self):
        self.time = None
        self.start_time = None
        self.start_iter = None

    def after_run(self, runner):
        self.start_time = time.time()
        self.start_iter = runner.iter

    def before_epoch(self, runner):
        self.time = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({"data_time": time.time() - self.time})

    def after_iter(self, runner):
        runner.log_buffer.update({"time": time.time() - self.time})
        self.time = time.time()

    def after_epoch(self, runner):
        runner.log_buffer.output["eta"] = (time.time() - self.start_time) / ()
