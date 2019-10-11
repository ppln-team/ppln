from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cifar.dataset import CustomCIFAR10
from ppln.data.transforms import build_albumentations
from ppln.hooks import ApexOptimizerHook
from ppln.utils.misc import get_dist_info


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


def batch_processor(model, data, mode):
    img, label = data
    label = label.cuda(non_blocking=True)
    pred = model(img.cuda(non_blocking=True))
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))

    values = OrderedDict()
    values['loss'] = loss.item()
    values['acc_top1'] = acc_top1.item()
    values['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, values=values, num_samples=img.size(0))
    return outputs


def build_dataset(data_root, transforms, train):
    transform = build_albumentations(transforms)
    return CustomCIFAR10(root=data_root, train=train, transform=transform)


def build_dataloader(dataset, images_per_gpu, workers_per_gpu, **kwargs):
    rank, world_size = get_dist_info()
    sampler = DistributedSampler(dataset, world_size, rank)
    batch_size = images_per_gpu
    num_workers = workers_per_gpu
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, **kwargs
    )


def build_data(data_root, train_transforms, val_transforms, images_per_gpu, workers_per_gpu):
    train_dataset = build_dataset(data_root=data_root, transforms=train_transforms, train=True)
    val_dataset = build_dataset(data_root=data_root, transforms=val_transforms, train=False)

    train_loader = build_dataloader(train_dataset, images_per_gpu, workers_per_gpu)
    val_loader = build_dataloader(val_dataset, images_per_gpu, workers_per_gpu)
    return train_loader, val_loader


def build_default_model(model, sync_bn):
    if sync_bn:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    return model


def build_apex(
    model, optimizer, optimizer_config, sync_bn, opt_level, keep_batchnorm_fp32, loss_scale, delay_allreduce
):
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    model, optimizer = amp.initialize(
        model, optimizer, opt_level=opt_level, keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale=loss_scale
    )
    if sync_bn:
        model = convert_syncbn_model(model)
    model = ApexDDP(model, delay_allreduce=delay_allreduce)
    optimizer_hook = ApexOptimizerHook(**optimizer_config)
    return model, optimizer, optimizer_hook
