import os
from logging import Logger
from typing import Any, Dict, List, Optional, Union

import torch

from .batch_processor import BaseBatchProcessor
from .factory import make_logger
from .hook_list import HookList
from .hooks import BaseHook
from .utils.checkpoint import load_checkpoint
from .utils.log_buffer import LogBuffer


class Runner(HookList):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        batch_processor: BaseBatchProcessor,
        hooks: List[Union[Dict, BaseHook]],
        work_dir: str,
        logger: Optional[Logger] = None
    ):
        super().__init__(hooks)
        self.work_dir = self.init_work_dir(work_dir)
        self.logger = self.init_logger(logger)
        self.optimizers = self.init_optimizers(optimizers)
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

    @staticmethod
    def init_optimizers(optimizers):
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = {'base': optimizers}
        return optimizers

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

    def run(self, data_loaders, max_epochs, resume_from=None, load_from=None, **kwargs):
        """Start running"""
        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

        self.max_epochs = max_epochs

        self.call('before_run')
        for self.epoch in range(self.epoch, max_epochs):
            for self.mode, self.data_loader in data_loaders.items():
                self.run_mode(**kwargs)
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

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load(checkpoint, map_location=map_location)

        self.epoch = checkpoint['meta']['epoch']
        self.iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            for name in self.optimizers:
                self.optimizers[name].load_state_dict(checkpoint['optimizer'][name])
        self.logger.info(f'resumed epoch {self.epoch}, iter {self.iter}')

    def load(self, filename, map_location: Any = 'cpu', strict: bool = False):
        self.logger.info(f'load checkpoint from {filename}')
        return load_checkpoint(self.model, filename, map_location, strict)
