from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class LogBufferHook(BaseHook):
    @property
    def priority(self):
        return Priority.LOW

    def after_iter(self, runner):
        runner.log_buffer.update(runner.batch_outputs["values"], runner.batch_outputs["num_samples"])
        runner.epoch_outputs.update(runner.log_buffer.average())

    def after_epoch(self, runner):
        runner.log_buffer.synchronize_between_processes()
        runner.epoch_outputs.update(runner.log_buffer.average())

    def before_epoch(self, runner):
        runner.log_buffer.clear()
        runner.epoch_outputs.clear()
