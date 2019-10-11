import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cifar.dataset import CustomCIFAR10
from ppln.data.transforms import build_albumentations
from ppln.hooks import ApexOptimizerHook
from ppln.utils.misc import get_dist_info


class BuildFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_dataset(self, mode):
        transforms = self.cfg.transforms[mode]
        transform = build_albumentations(transforms)
        return CustomCIFAR10(root=self.cfg.data.data_root, train=mode == 'train', transform=transform)

    def build_dataloader(self, dataset):
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(dataset, world_size, rank)
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.images_per_gpu,
            sampler=sampler,
            num_workers=self.cfg.data.workers_per_gpu,
            pin_memory=False
        )

    def build_data(self):
        train_dataset = self.build_dataset(mode='train')
        val_dataset = self.build_dataset(mode='val')

        train_loader = self.build_dataloader(train_dataset)
        val_loader = self.build_dataloader(val_dataset)
        return train_loader, val_loader

    def build_default_model(self, model):
        if self.cfg.sync_bn:
            model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        return model

    def build_apex(self, model, optimizer):
        from apex import amp
        from apex.parallel import DistributedDataParallel as ApexDDP
        from apex.parallel import convert_syncbn_model

        delay_allreduce = self.cfg.apex.pop('delay_allreduce')
        model, optimizer = amp.initialize(model, optimizer, **self.cfg.apex)
        if self.cfg.sync_bn:
            model = convert_syncbn_model(model)
        model = ApexDDP(model, delay_allreduce=delay_allreduce)
        optimizer_hook = ApexOptimizerHook(**self.cfg.optimizer_config)
        return model, optimizer, optimizer_hook
