import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.nn import SyncBatchNorm
from torchvision.models import resnet

from ppln.utils.config import Config
from ppln.hooks import DistSamplerSeedHook
from ppln.runner import Runner


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode, device):
    assert isinstance(train_mode, bool)
    img, label = data
    label = label.to(device, non_blocking=True)
    pred = model(img)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))

    values = OrderedDict()
    values['loss'] = loss.item()
    values['acc_top1'] = acc_top1.item()
    values['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, values=values, num_samples=img.size(0))
    return outputs


def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    torch.distributed.init_process_group(backend=backend, **kwargs)


def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
    else:
        dist = True
        init_dist(**cfg.dist_params)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # build datasets and dataloaders
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    train_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    )
    val_dataset = datasets.CIFAR10(
        root=cfg.data_root, train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    if dist:
        num_workers = cfg.data_workers
        assert cfg.batch_size % world_size == 0
        batch_size = cfg.batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, world_size, rank)
        val_sampler = DistributedSampler(val_dataset, world_size, rank)
        shuffle = False
    else:
        num_workers = cfg.data_workers * len(cfg.gpus)
        batch_size = cfg.batch_size
        train_sampler = None
        val_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=False
    )

    # build model
    model = getattr(resnet, cfg.model)(pretrained=True)
    if dist:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model, device_ids=cfg.gpus)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build runner and register hooks
    runner = Runner(model, cfg.optimizer, batch_processor, device, cfg.work_dir)
    runner.register_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config
    )
    if dist:
        runner.register_hook(DistSamplerSeedHook())

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load(cfg.load_from)

    runner.run({'train': train_loader, 'val': val_loader}, cfg.total_epochs)


if __name__ == '__main__':
    main()
