import os
import os.path as osp
import time
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from .. import __version__
from .misc import get_dist_info


def load_state_dict(module, state_dict, strict=False):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)
        except Exception:
            raise RuntimeError(
                'While copying the parameter named {}, '
                'whose dimensions in the model are {} and '
                'whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size())
            )

    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    rank, _ = get_dist_info()
    if err_msg and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)


def load_optim_state_dict(obj, state_dict, obj_type):
    if isinstance(obj, dict):
        for name in obj:
            obj[name].load_state_dict(state_dict[name])
    elif isinstance(obj, obj_type):
        obj.load_state_dict(state_dict)


def load_checkpoint(model, filename, map_location=None, strict=False, optimizer=None, scheduler=None):
    """Load checkpoint from a file or URI."""
    checkpoint = torch.load(filename, map_location=map_location)

    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)

    if 'optimizer' in checkpoint and optimizer is not None:
        load_optim_state_dict(optimizer, checkpoint['optimizer'], Optimizer)
    if 'scheduler' in checkpoint and scheduler is not None:
        load_optim_state_dict(scheduler, checkpoint['scheduler'], _LRScheduler)

    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def make_optim_state_dict(data, data_type):
    if isinstance(data, dict):
        return {k: v.state_dict() for k, v in data.items()}
    elif isinstance(data, data_type):
        return data.state_dict()


def save_checkpoint(model, filename, optimizer=None, scheduler=None, meta=None):
    """Save checkpoint to file.
    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain __version__.py and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        scheduler (:obj:`Scheduler`, optional): Scheduler to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(type(meta)))
    meta.update(ppln_version=__version__, time=time.asctime())

    os.makedirs(osp.dirname(filename), exist_ok=True)
    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict()),
    }
    if optimizer is not None:
        checkpoint['optimizer'] = make_optim_state_dict(optimizer, Optimizer)
    if scheduler is not None:
        checkpoint['scheduler'] = make_optim_state_dict(scheduler, _LRScheduler)
    torch.save(checkpoint, filename)
