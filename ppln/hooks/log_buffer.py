import os
import os.path as osp
import torch.distributed as dist

from ..fileio import io
from .hook import Hook


class LogBufferHook(Hook):
    def __init__(self, distributed=True):
        self.distributed = distributed

    @staticmethod
    def sync(runner):
        if runner.rank == 0:
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, f'tmp_{i}.pkl')
                tmp_results = io.load(tmp_file)
                n_history = tmp_results['n_history']
                value_history = tmp_results['value_history']
                for key in n_history:
                    runner.log_buffer.value_history[key].extend(value_history[key])
                    runner.log_buffer.n_history[key].extend(n_history[key])
                os.remove(tmp_file)
        else:
            tmp_file = osp.join(runner.work_dir, f'tmp_{runner.rank}.pkl')
            io.dump(
                {
                    'value_history': runner.log_buffer.value_history,
                    'n_history': runner.log_buffer.value_history
                }, tmp_file
            )
            dist.barrier()
        dist.barrier()

    def before_epoch(self, runner):
        runner.log_buffer.clear()

    def after_train_iter(self, runner):
        runner.log_buffer.average()

    def after_train_epoch(self, runner):
        if self.distributed:
            self.sync(runner)
        runner.log_buffer.average()

    def after_val_iter(self, runner):
        if self.distributed:
            self.sync(runner)
        runner.log_buffer.average()
