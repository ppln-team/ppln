from collections import OrderedDict

import torch
import torch.nn.functional as F


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


def make_test_output(pred, data):
    return dict(
        values=torch.argmax(pred, dim=1).cpu().numpy(),
        num_samples=data['image'].size(0),
        index=data['index'].numpy(),
        gt_label=data['target'].numpy()
    )


def make_train_output(pred, data):
    label = data['target'].cuda(non_blocking=True)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))

    values = OrderedDict()
    values['loss'] = loss.item()
    values['acc_top1'] = acc_top1.item()
    values['acc_top5'] = acc_top5.item()
    return dict(loss=loss, values=values, num_samples=data['image'].size(0))


def batch_processor(model, data, mode):
    pred = model(data['image'].cuda(non_blocking=True))
    if mode == 'test':
        return make_test_output(pred, data)
    else:
        return make_train_output(pred, data)
