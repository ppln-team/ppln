import os
import os.path as osp
from queue import PriorityQueue
import numpy as np

from .hook import Hook
from ..utils.misc import master_only
from ..utils.checkpoint import save_checkpoint


class CheckpointHook(Hook):
    def __init__(self, metric_name, mode, num_checkpoints=5, save_optimizer=True, out_dir=None, **kwargs):
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.meta = kwargs
        self._checkpoints = PriorityQueue(num_checkpoints)
        self._metric_name = metric_name
        self._best_metric = (-np.infty, +np.infty)[mode == 'min']

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

    def current_filename(self, runner):
        return osp.join(self.out_dir, f'epoch_{runner.epoch + 1}.pth')

    @master_only
    def after_val_epoch(self, runner):
        metric = runner.log_buffer.output[self._metric_name]
        if self._is_update(metric):
            self._checkpoints.put((metric, self.current_filename(runner)))
            self._save_checkpoint(runner)
        if self._cmp(self._best_metric, metric):
            self._best_metric = metric
            self._save_link(runner)

    def _cmp(self, x, y):
        return (x < y and self.mode == 'max') or (x > y and self.mode == 'min')

    def _is_update(self, metric):
        if not self._checkpoints.full():
            return True

        min_metric, min_filename = self._checkpoints.get()

        if self._cmp(min_metric, metric):
            os.remove(min_filename)
            return True

        self._checkpoints.put((min_metric, min_filename))
        return False

    def _save_link(self, runner):
        linkname = osp.join(self.out_dir, 'best.pth')
        if os.path.lexists(linkname):
            os.remove(linkname)
        os.symlink(self.current_filename(runner), linkname)

    def _save_checkpoint(self, runner):
        self.meta.update(epoch=runner.epoch + 1, iter=runner.iter)

        optimizer = runner.optimizer if self.save_optimizer else None
        save_checkpoint(runner.model, self.current_filename(runner), optimizer=optimizer, meta=self.meta)
