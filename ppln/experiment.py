import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict
from .data.transforms import make_albumentations
from .factory import make_model, make_scheduler
from .hooks import DistSamplerSeedHook, IterTimerHook, LogBufferHook
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._optimizers = None

    @property
    def optimizers(self) -> Dict[str, Optimizer]:
        raise NotImplementedError

    @property
    def schedulers(self) -> Dict[str, _LRScheduler]:
        schedulers = {}
        for name, optimizer in self.optimizers:
            schedulers[name] = make_scheduler(optimizer, self.cfg.schedulers[name])
        return schedulers

    @property
    def model(self):
        if self._model is None:
            self._model = make_model(self.cfg.model)
        return self._model

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
