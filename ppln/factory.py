import logging
import os.path as osp
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel as PytorchDDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from .utils.misc import get_dist_info, get_timestamp, object_from_dict

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
except ImportError as e:
    warnings.warn(
        f"Error \"{e}\" during importing apex library. To use mixed precison"
        ' you should install it from https://github.com/NVIDIA/apex'
    )


def make_model(cfg) -> torch.nn.Module:
    model = object_from_dict(cfg)
    return model.cuda()


def make_pytorch_ddp(model, **kwargs) -> torch.nn.Module:
    return PytorchDDP(model, device_ids=[torch.cuda.current_device()], **kwargs)


def make_apex_ddp(model, **kwargs) -> torch.nn.Module:
    return ApexDDP(model, **kwargs)


def make_apex(model, optimizer=None, **kwargs):
    if optimizer is None:
        model = amp.initialize(model, **kwargs)
        return model
    else:
        model, optimizer = amp.initialize(model, optimizer, **kwargs)
        return model, optimizer


def make_optimizer(model: torch.nn.Module, config: dict) -> Optimizer:
    return object_from_dict(config, params=filter(lambda x: x.requires_grad, model.parameters()))


def make_scheduler(optimizer: Optimizer, config: dict) -> _LRScheduler:
    return object_from_dict(config, optimizer=optimizer)


def make_file_handler(logger, filename=None, mode='w', level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def make_logger(log_dir):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    rank, _ = get_dist_info()
    timestamp = get_timestamp()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if log_dir and rank == 0:
        log_file = osp.join(log_dir, f'{timestamp}.log')
        make_file_handler(logger, log_file, level=logging.INFO)
    return logger
