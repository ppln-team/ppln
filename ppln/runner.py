import os

import torch

from . import hooks
from .factory import make_logger, make_model, make_optimizer
from .hooks import CheckpointHook, Hook, IterTimerHook, LogBufferHook, OptimizerHook, get_priority, lr_scheduler
from .utils.checkpoint import load_checkpoint
from .utils.log_buffer import LogBuffer
from .utils.misc import object_from_dict


class Runner:
    def __init__(self, model, optimizer, batch_processor, work_dir, logger=None):
        self.work_dir = self.init_work_dir(work_dir)
        self.logger = self.init_logger(logger)
        self.model = self.init_model(model)
        self.optimizer = self.init_optimizer(optimizer)

        self.batch_processor = batch_processor
        self.log_buffer = LogBuffer()

        self.hooks = []
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

    def run(self, data_loaders, max_epochs, **kwargs):
        """Start running"""
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
            self.outputs = self.batch_processor(self.model, batch, mode=self.mode, **kwargs)
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

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self.hooks) - 1, -1, -1):
            if priority >= self.hooks[i].priority:
                self.hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self.hooks.insert(0, hook)

    def register_hooks(self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None):
        """Register default hooks for training."""
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_hook(object_from_dict(lr_config, lr_scheduler))
        self.register_hook(
            optimizer_config if isinstance(optimizer_config, Hook) else OptimizerHook(**optimizer_config)
        )
        self.register_hook(IterTimerHook())
        self.register_hook(LogBufferHook(), priority='LOW')
        for hook_params in log_config['hooks']:
            logger_hook = object_from_dict(hook_params, hooks)
            self.register_hook(logger_hook, priority='VERY_LOW')
        self.register_hook(CheckpointHook(**checkpoint_config), priority='LOWEST')
