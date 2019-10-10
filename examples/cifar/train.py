from argparse import ArgumentParser

import torch
from apex import amp
from torchvision.models import resnet

from cifar.utils import batch_processor, build_dataloader, build_dataparallel, build_dataset
from ppln.factory import make_optimizer
from ppln.hooks import DistSamplerSeedHook
from ppln.hooks.apex import ApexOptimizerHook
from ppln.runner import Runner
from ppln.utils.config import Config
from ppln.utils.misc import init_dist


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
    init_dist(**cfg.dist_params)

    # build datasets and dataloaders
    train_dataset = build_dataset(data_root=cfg.data_root, transforms=cfg.val_transforms, train=True)
    val_dataset = build_dataset(data_root=cfg.data_root, transforms=cfg.train_transforms, train=False)

    train_loader = build_dataloader(train_dataset, cfg.images_per_gpu, cfg.workers_per_gpu)
    val_loader = build_dataloader(val_dataset, cfg.images_per_gpu, cfg.workers_per_gpu)

    # build model
    model = getattr(resnet, cfg.model)(pretrained=True).cuda()
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
        optimizer_hook = ApexOptimizerHook(cfg.optimizer_config.grad_clip)
    else:
        optimizer = cfg.optimizer
        optimizer_hook = cfg.optimizer_config

    model = build_dataparallel(
        model,
        apex=cfg.apex,
        sync_bn=cfg.sync_bn,
        delay_allreduce=cfg.delay_allreduce,
        device_ids=[torch.cuda.current_device()]
    )

    # build runner and register hooks
    runner = Runner(model, optimizer, batch_processor, cfg.work_dir)
    runner.register_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=optimizer_hook,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config
    )
    runner.register_hook(DistSamplerSeedHook())

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load(cfg.load_from)

    runner.run({'train': train_loader, 'val': val_loader}, cfg.total_epochs)


if __name__ == '__main__':
    main()
