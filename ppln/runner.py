import os

import torch

from . import hooks as _hooks
from .factory import make_logger, make_model, make_optimizer
from .hooks import Hook, IterTimerHook, LogBufferHook
from .utils.checkpoint import load_checkpoint
from .utils.log_buffer import LogBuffer
from .utils.misc import object_from_dict


class Runner:
    def __init__(self, model, optimizer, batch_processor, hooks, work_dir, logger=None):
        self.work_dir = self.init_work_dir(work_dir)
        self.logger = self.init_logger(logger)
        self.model = self.init_model(model)
        self.optimizer = self.init_optimizer(optimizer)
        self.hooks = []

        self.batch_processor = batch_processor
        self.log_buffer = LogBuffer()
        self.register_hooks(hooks)

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

    def call_hook(self, action):
        for hook in self.hooks:
            getattr(hook, action)(self)

    def run(self, data_loaders, max_epochs, resume_from=None, load_from=None, **kwargs):
        """Start running"""
        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

        self.max_epochs = max_epochs

        self.call_hook('before_run')
        for self.epoch in range(self.epoch, max_epochs):
            for self.mode, self.data_loader in data_loaders.items():
                self.run_mode(**kwargs)
        self.call_hook('after_run')

    def run_mode(self, **kwargs):
        self.model.train(self.train_mode)

        if self.train_mode:
            self.max_iters = self.max_epochs * len(self.data_loader)

        self.call_hook(f'before_{self.mode}_epoch')
        for self.inner_iter, batch in enumerate(self.data_loader):
            self.run_batch(batch, **kwargs)
        self.call_hook(f'after_{self.mode}_epoch')

    def run_batch(self, batch, **kwargs):
        self.call_hook(f'before_{self.mode}_iter')
        with torch.set_grad_enabled(self.train_mode):
            self.outputs = getattr(self.batch_processor, f'{self.mode}_step')(self.model, batch, **kwargs)
            self.log_buffer.update(self.outputs['values'], self.outputs['num_samples'])
        self.call_hook(f'after_{self.mode}_iter')

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

    def register_hook(self, hook):
        """Register a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
        """
        if isinstance(hook, dict):
            hook = object_from_dict(hook, _hooks)

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self.hooks) - 1, -1, -1):
            if hook.priority >= self.hooks[i].priority:
                self.hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self.hooks.insert(0, hook)

    def register_hooks(self, hooks):
        for hook in hooks:
            self.register_hook(hook if isinstance(hook, Hook) else object_from_dict(hook, _hooks))
        self.register_hook(IterTimerHook())
        self.register_hook(LogBufferHook())
