import mlflow

from ...utils.misc import master_only
from ..priority import Priority
from ..registry import HOOKS
from .logger import BaseLoggerHook
from .utils import get_lr


@HOOKS.register_module
class MlFlowLoggerHook(BaseLoggerHook):
    def __init__(self):
        super().__init__()

    @property
    def priority(self):
        return Priority.VERY_LOW

    @master_only
    def log_lr(self, lr_dict):
        for optimizer_name, lrs in lr_dict.items():
            if len(lrs) == 1:
                mlflow.log_metric(f'{optimizer_name}_lr', lrs[0])
            else:
                for i, lr in enumerate(lrs):
                    mlflow.log_metric(f'{optimizer_name}_{i}_lr', lr)

    @master_only
    def log(self, runner):
        for value in runner.log_buffer.output:
            if value in ['time', 'data_time']:
                continue
            tag = f'{runner.mode}_{value}'
            record = runner.log_buffer.output[value]
            if isinstance(record, str):
                pass
            else:
                mlflow.log_metric(tag, record, runner.epoch)

            if runner.mode == 'train':
                lr_dict = get_lr(runner.optimizers)
                self.log_lr(lr_dict)

    def after_epoch(self, runner):
        self.log(runner)
