import pickle
from collections import defaultdict

import torch
import torch.distributed as dist

from ppln.utils.misc import get_dist_info
from ppln.utils.progress_bar import ProgressBar

MAX_LEN = 512


def multi_gpu_test(model, data_loader, batch_processor, **kwargs):
    model.eval()
    results = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        bar = ProgressBar(len(dataset))
    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            result = batch_processor.test_step(model, batch, **kwargs)
            num_samples = result.pop('num_samples')
        results.append(result)

        if rank == 0:
            for _ in range(num_samples * world_size):
                bar.update()
    return results


def collect_results(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max().item()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = [pickle.loads(t[:shape_list[i][0]].cpu().numpy().tobytes()) for i, t in enumerate(part_recv_list)]
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]

        results = defaultdict(list)
        for res in ordered_results:
            for k, v in res.items():
                results[k].extend(list(v))

        return results
