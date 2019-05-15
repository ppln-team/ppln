import torch
import logging

from .utils.misc import object_from_dict


def make_model(config, device) -> torch.nn.Module:
    model = object_from_dict(config)
    return model.to(device)


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer = object_from_dict(config, params=filter(lambda x: x.requires_grad, model.parameters()))
    return optimizer


def make_file_handler(logger, filename=None, mode='w', level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger
