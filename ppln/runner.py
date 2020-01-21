import os
from logging import Logger
from typing import Dict, List, Optional, Union

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer

from .batch_processor import BaseBatchProcessor
from .factory import make_logger
from .hook_list import HookList
from .hooks import BaseHook
from .utils.log_buffer import LogBuffer


class Runner(HookList):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Union[Optimizer, Dict[str, Optimizer]],
        schedulers: Union[_LRScheduler, Dict[str, _LRScheduler]],
        batch_processor: BaseBatchProcessor,
        hooks: List[Union[Dict, BaseHook]],
        work_dir: str,
        logger: Optional[Logger] = None
    ):
        super().__init__(hooks)
        self.work_dir = self.init_work_dir(work_dir)
        self.logger = self.init_logger(logger)
        self.optimizers = self.init_optimizers(optimizers)
        self.schedulers = self.init_schedulers(schedulers)
        self.model = model

        self.batch_processor = batch_processor
        self.log_buffer = LogBuffer()

        self.outputs = None
        self.mode = None
        self.data_loader = None
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.max_epochs = 0
        self.max_iters = 0
        self.stop_training = False

    @staticmethod
    def init_optimizers(optimizers):
        if isinstance(optimizers, Optimizer):
            optimizers = {'base': optimizers}
        return optimizers

    @staticmethod
    def init_schedulers(schedulers):
        if isinstance(schedulers, (_LRScheduler, ReduceLROnPlateau)):
            schedulers = {'base': schedulers}
        return schedulers

    @staticmethod
    def init_work_dir(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def init_logger(self, logger):
        if logger is None:
            return make_logger(self.work_dir)
        return logger

    @property
    def train_mode(self) -> bool:
        return self.mode == 'train'

    def run(self, data_loaders, max_epochs, **kwargs):
        """Start running"""
        self.max_epochs = max_epochs

        self.call('before_run')
        for self.epoch in range(self.epoch, max_epochs):
            for self.mode, self.data_loader in data_loaders.items():
                self.run_mode(**kwargs)
            if self.stop_training:
                break
        self.call('after_run')

    def run_mode(self, **kwargs):
        self.model.train(self.train_mode)

        if self.train_mode:
            self.max_iters = self.max_epochs * len(self.data_loader)

        self.call(f'before_{self.mode}_epoch')
        for self.inner_iter, batch in enumerate(self.data_loader):
            self.run_batch(batch, **kwargs)
        self.call(f'after_{self.mode}_epoch')

    def run_batch(self, batch, **kwargs):
        self.call(f'before_{self.mode}_iter')
        with torch.set_grad_enabled(self.train_mode):
            self.outputs = getattr(self.batch_processor, f'{self.mode}_step')(self.model, batch, **kwargs)
            self.log_buffer.update(self.outputs['values'], self.outputs['num_samples'])
        self.call(f'after_{self.mode}_iter')

        if self.train_mode:
            self.iter += 1
