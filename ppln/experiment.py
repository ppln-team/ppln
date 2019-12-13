import torch.distributed as dist
from torch.utils.data import DataLoader

from .data.transforms import make_albumentations
from .factory import make_model, make_optimizer
from .hooks import DistSamplerSeedHook, IterTimerHook, LogBufferHook


class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    @property
    def optimizer(self):
        return make_optimizer(self.model, self.cfg.optimizer)

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
