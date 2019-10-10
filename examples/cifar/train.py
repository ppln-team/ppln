from argparse import ArgumentParser

import torch
from torchvision.models import resnet

from cifar.utils import batch_processor, build_dataloader, build_dataparallel, build_dataset
from ppln.hooks import DistSamplerSeedHook
from ppln.runner import Runner
from ppln.utils.config import Config
from ppln.utils.misc import init_dist
from ppln.hooks.apex import ApexOptimizerHook
from ppln.factory import make_optimizer
from apex import amp


def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    num_gpus = torch.cuda.device_count()

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
    else:
        init_dist(**cfg.dist_params)
        dist = True

    # build datasets and dataloaders
    train_dataset = build_dataset(data_root=cfg.data_root, transforms=cfg.val_transforms, train=True)
    val_dataset = build_dataset(data_root=cfg.data_root, transforms=cfg.train_transforms, train=False)

    train_loader = build_dataloader(
        train_dataset, cfg.images_per_gpu, cfg.workers_per_gpu, num_gpus=num_gpus, dist=dist, shuffle=True
    )
    val_loader = build_dataloader(
        val_dataset, cfg.images_per_gpu, cfg.workers_per_gpu, num_gpus=num_gpus, dist=dist, shuffle=False
    )

    # build model
    model = getattr(resnet, cfg.model)(pretrained=True)

    optimizer = make_optimizer(model, cfg.optimizer)

    # apex
    if cfg.apex:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=cfg.opt_level,
            keep_batchnorm_fp32=cfg.keep_batchnorm_fp32,
            loss_scale=cfg.loss_scale
        )

    model = build_dataparallel(model, dist=dist, sync_bn=cfg.sync_bn, apex=cfg.apex)

    # build runner and register hooks
    runner = Runner(model, cfg.optimizer, batch_processor, cfg.work_dir)
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
