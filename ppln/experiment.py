from torch.utils.data import DataLoader

from ppln.data.transforms import build_albumentations
from ppln.factory import make_apex, make_ddp, make_model, make_optimizer


class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = make_model(self.cfg.model)
        self.optimizer = make_optimizer(self.model, self.cfg.optimizer)
        if self.cfg.apex is not None:
            self.model, self.optimizer = make_apex(self.cfg.apex, self.model, self.optimizer)
        elif self.cfg.ddp is not None:
            self.model = make_ddp(self.model)

    def transform(self, mode):
        return build_albumentations(self.cfg.transforms[mode])

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
