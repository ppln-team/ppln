import argparse

import torch
from torchvision.models import resnet

from cifar.builder import BuildFactory
from cifar.utils import batch_processor
from ppln.fileio import io
from ppln.inference import collect_results, multi_gpu_test
from ppln.utils.checkpoint import load_checkpoint
from ppln.utils.config import Config
from ppln.utils.misc import init_dist


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.sync_bn = False

    init_dist(**cfg.dist_params)

    # build data
    builder = BuildFactory(cfg)
    dataset = builder.build_dataset('val')
    data_loader = builder.build_dataloader(dataset)

    # build model
    model = getattr(resnet, cfg.model)(pretrained=True).cuda()
    model = builder.build_default_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    outputs = multi_gpu_test(model, data_loader, batch_processor)
    outputs = collect_results(outputs, len(data_loader.dataset))
    io.dump(outputs, args.out)


if __name__ == '__main__':
    main()
