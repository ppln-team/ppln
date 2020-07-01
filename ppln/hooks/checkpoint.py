import os
import os.path as osp
from queue import PriorityQueue

import numpy as np

from ..utils.checkpoint import save_checkpoint
from ..utils.dist import master_only
from .base import BaseHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class CheckpointHook(BaseHook):
    def __init__(
        self, monitor_metric="loss", mode="min", num_checkpoints=5, save_optimizer=True, save_scheduler=True, **kwargs
    ):
        self.monitor_metric = monitor_metric
        self.meta = kwargs
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self._checkpoints = PriorityQueue(num_checkpoints)
        if mode == "min":
            self._compare_metrics = np.less
            self._best_metric = np.infty
        elif mode == "max":
            self._compare_metrics = np.greater
            self._best_metric = -np.infty

    @property
    def priority(self):
        return Priority.VERY_LOW

    @staticmethod
    def current_filename(runner):
        return f"epoch_{runner.epoch + 1}.pth"

    def current_filepath(self, runner):
        return osp.join(runner.work_dir, self.current_filename(runner))

    @master_only
    def after_val_epoch(self, runner):
        metric = runner.epoch_outputs[self.monitor_metric]

        if self._is_update(metric):
            self._checkpoints.put((metric, self.current_filepath(runner)))
            self._save_checkpoint(runner)
        if self._compare_metrics(metric, self._best_metric):
            self._best_metric = metric
            self._save_link(runner)
            runner.logger.info(
                f"Best checkpoint was changed: {self.current_filename(runner)} with {self._best_metric}"
            )

    @master_only
    def after_run(self, runner):
        runner.logger.info("Best checkpoints:")
        while not self._checkpoints.empty():
            metric, filename = self._checkpoints.get()
            runner.logger.info(f"{filename}: {metric}")

    def _is_update(self, metric):
        if not self._checkpoints.full():
            return True

        min_metric, min_filename = self._checkpoints.get()
        if self._compare_metrics(metric, min_metric):
            os.remove(min_filename)
            return True

        self._checkpoints.put((min_metric, min_filename))
        return False

    def _save_link(self, runner):
        linkpath = osp.join(runner.work_dir, "best.pth")
        if os.path.lexists(linkpath):
            os.remove(linkpath)
        os.symlink(self.current_filename(runner), linkpath)

    def _save_checkpoint(self, runner):
        self.meta.update(epoch=runner.epoch + 1, iter=runner.iter)

        optimizer = runner.optimizer if self.save_optimizer else None
        scheduler = runner.scheduler if self.save_scheduler else None
        save_checkpoint(
            runner.model, self.current_filepath(runner), optimizer=optimizer, scheduler=scheduler, meta=self.meta
        )
