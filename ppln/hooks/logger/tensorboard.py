from torch.utils.tensorboard import SummaryWriter

from ...utils.misc import master_only
from .. import BaseHook
from ..priority import Priority
from ..registry import HOOKS


@HOOKS.register_module
class TensorboardLoggerHook(BaseHook):
    def __init__(self, log_dir=None):
        super(TensorboardLoggerHook, self).__init__()
        self.log_dir = log_dir
        self.writer = None

    @property
    def priority(self):
        return Priority.VERY_LOW

    @master_only
    def before_run(self, runner):
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for value in runner.log_buffer.output:
            if value in ['time', 'data_time']:
                continue
            tag = f'{value}/{runner.mode}'
            record = runner.log_buffer.output[value]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, record, runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def after_epoch(self, runner):
        self.log(runner)
