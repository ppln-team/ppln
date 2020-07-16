from collections import defaultdict

import torch

from ppln.utils.dist import all_gather_cpu, all_gather_gpu, get_dist_info
from ppln.utils.progress_bar import ProgressBar


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
            num_samples = result.pop("num_samples")
        results.append(result)

        if rank == 0:
            for _ in range(num_samples * world_size):
                bar.update()
    return results


def order_part_list(part_list, size):
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the data loader may pad some samples
    ordered_results = ordered_results[:size]

    results = defaultdict(list)
    for res in ordered_results:
        for k, v in res.items():
            results[k].extend(list(v))

    return results


def collect_results(result_part, size, gpu_collect=False):
    rank, world_size = get_dist_info()
    if gpu_collect:
        part_list = all_gather_gpu(result_part)
    else:
        part_list = all_gather_cpu(result_part)
    if rank == 0:
        return order_part_list(part_list, size)
