import os.path as osp

from ...utils.misc import master_only
from .. import BaseHook
from ..priority import Priority


class TensorboardLoggerHook(BaseHook):
    def __init__(self, log_dir=None, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(reset_flag)
        self.log_dir = log_dir

    @property
    def priority(self):
        return Priority.VERY_LOW

    @master_only
    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'tf_logs')
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
