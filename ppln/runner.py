import torch
import logging
import os
import os.path as osp
from .utils.misc import object_from_dict, get_dist_info, get_timestamp
from .utils.log_buffer import LogBuffer
from .utils.checkpoint import load_checkpoint
from .factory import make_model, make_optimizer, make_file_handler

from . import hooks
from .hooks import Hook, get_priority, OptimizerHook, CheckpointHook, IterTimerHook, lr_scheduler, LogBufferHook


class Runner:
    def __init__(self, model, optimizer, batch_processor, device, work_dir):
        self.work_dir = osp.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

        self.device = device
        self.work_dir = work_dir
        self.batch_processor = batch_processor

        self.log_buffer = LogBuffer()
        self.rank, self.world_size = get_dist_info()
        self.timestamp = get_timestamp()

        self.logger = self.init_logger(work_dir)
        self.model = self.init_model(model)
        self.optimizer = self.init_optimizer(optimizer)

        self.data_loader = None
        self.outputs = None
        self.mode = None
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.max_epochs = 0
        self.max_iters = 0

    def init_model(self, model):
        if isinstance(model, dict):
            return make_model(model, self.device)
        else:
            return model

    def init_optimizer(self, optimizer):
        if isinstance(optimizer, dict):
            return make_optimizer(self.model, optimizer)
        else:
            return optimizer

    def init_logger(self, log_dir=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if log_dir and self.rank == 0:
            log_file = osp.join(log_dir, f'{self.timestamp}.log')
            make_file_handler(logger, log_file, level=logging.INFO)
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
        for self.epoch in range(max_epochs):
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
            self.outputs = self.batch_processor(
                self.model, batch, train_mode=self.train_mode, device=self.device, **kwargs
            )
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
        self.register_hook(OptimizerHook(**optimizer_config))
        self.register_hook(IterTimerHook())
        self.register_hook(LogBufferHook(), priority='LOW')
        for hook_params in log_config['hooks']:
            logger_hook = object_from_dict(hook_params, hooks)
            self.register_hook(logger_hook, priority='VERY_LOW')
        self.register_hook(CheckpointHook(**checkpoint_config), priority='LOWEST')
