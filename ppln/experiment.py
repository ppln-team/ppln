from typing import Dict

import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .data.transforms import make_albumentations
from .factory import make_model
from .hooks import DistSamplerSeedHook, IterTimerHook, LogBufferHook
from .utils.misc import cached_property


class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg

    @cached_property
    def optimizers(self) -> Dict[str, Optimizer]:
        raise NotImplementedError

    @property
    def schedulers(self) -> Dict[str, _LRScheduler]:
        raise NotImplementedError

    @cached_property
    def model(self) -> nn.Module:
        return make_model(self.cfg.model)

    def transform(self, mode):
        return make_albumentations(self.cfg.transforms[mode])

    def dataset(self, mode):
        raise NotImplementedError

    def sampler(self, mode, dataset):
        raise NotImplementedError

    def dataloader(self, mode):
        dataset = self.dataset(mode)
        sampler = self.sampler(mode, dataset)
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.cfg.data.images_per_gpu,
            num_workers=self.cfg.data.workers_per_gpu,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=mode == 'train'
        )

    @property
    def hooks(self):
        hooks = self.cfg['hooks']
        if dist.is_initialized():
            hooks.append(DistSamplerSeedHook())
        return self.cfg['hooks'] + [IterTimerHook(), LogBufferHook()]

    @property
    def work_dir(self):
        return self.cfg.work_dir
