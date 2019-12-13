import os

import torch

from .factory import make_logger, make_model, make_optimizer
from .hook_list import HookList
from .utils.checkpoint import load_checkpoint
from .utils.log_buffer import LogBuffer


class Runner:
    def __init__(self, model, optimizer, batch_processor, hooks, work_dir, logger=None):
        self.work_dir = self.init_work_dir(work_dir)
        self.logger = self.init_logger(logger)
        self.model = self.init_model(model)
        self.optimizer = self.init_optimizer(optimizer)
        self.hook_list = HookList(hooks)

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
    def init_work_dir(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    @staticmethod
    def init_model(model):
        if isinstance(model, dict):
            return make_model(model)
        return model

    def init_optimizer(self, optimizer):
        if isinstance(optimizer, dict):
            return make_optimizer(self.model, optimizer)
        return optimizer

    def init_logger(self, logger):
        if logger is None:
            return make_logger(self.work_dir)
        return logger

    @property
    def lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError('lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    @property
    def train_mode(self):
        """bool: Train mode."""
        return self.mode == 'train'

    def run(self, data_loaders, max_epochs, resume_from=None, load_from=None, **kwargs):
        """Start running"""
        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

        self.max_epochs = max_epochs

        self.hook_list(self, 'before_run')
        for self.epoch in range(self.epoch, max_epochs):
            for self.mode, self.data_loader in data_loaders.items():
                self.run_mode(**kwargs)
        self.hook_list(self, 'after_run')

    def run_mode(self, **kwargs):
        self.model.train(self.train_mode)

        if self.train_mode:
            self.max_iters = self.max_epochs * len(self.data_loader)

        self.hook_list(self, f'before_{self.mode}_epoch')
        for self.inner_iter, batch in enumerate(self.data_loader):
            self.run_batch(batch, **kwargs)
        self.hook_list(self, f'after_{self.mode}_epoch')

    def run_batch(self, batch, **kwargs):
        self.hook_list(self, f'before_{self.mode}_iter')
        with torch.set_grad_enabled(self.train_mode):
            self.outputs = getattr(self.batch_processor, f'{self.mode}_step')(self.model, batch, **kwargs)
            self.log_buffer.update(self.outputs['values'], self.outputs['num_samples'])
        self.hook_list(self, f'after_{self.mode}_iter')

        if self.train_mode:
            self.iter += 1

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                self.model, checkpoint, map_location=lambda storage, loc: storage.cuda(device_id)
            )
        else:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location=map_location)

        self.epoch = checkpoint['meta']['epoch']
        self.iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def load(self, filename, map_location='cpu', strict=False):
        return load_checkpoint(self.model, filename, map_location, strict)
