import os
import os.path as osp

import torch
import torch.distributed as dist

from ..fileio import io
from ..utils.misc import get_dist_info
from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class LogBufferHook(BaseHook):
    @property
    def priority(self):
        return Priority.LOW

    @staticmethod
    def sync(runner):
        rank, world_size = get_dist_info()

        if rank == 0:
            dist.barrier()
            for i in range(1, world_size):
                tmp_file = osp.join(runner.work_dir, f'tmp_{i}.pkl')
                tmp_results = io.load(tmp_file)
                n_history = tmp_results['n_history']
                value_history = tmp_results['value_history']
                for key in n_history:
                    runner.log_buffer.value_history[key].extend(value_history[key])
                    runner.log_buffer.n_history[key].extend(n_history[key])
                os.remove(tmp_file)
        else:
            tmp_file = osp.join(runner.work_dir, f'tmp_{rank}.pkl')
            io.dump(
                {
                    'value_history': runner.log_buffer.value_history,
                    'n_history': runner.log_buffer.n_history
                }, tmp_file
            )
            dist.barrier()
        dist.barrier()

    def before_epoch(self, runner):
        runner.log_buffer.clear()

    def after_iter(self, runner):
        runner.log_buffer.average()

    def after_epoch(self, runner):
        if dist.is_initialized():
            self.sync(runner)
        runner.log_buffer.average()
        if dist.is_initialized():
            for key, value in runner.log_buffer.output.items():
                value_tensor = torch.tensor(value, device='cuda')
                dist.broadcast(value_tensor, 0)
                runner.log_buffer.output[key] = value_tensor.item()
