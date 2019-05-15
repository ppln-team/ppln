import torch
import torch.distributed as dist


def get_max_memory(runner):
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([mem / (1024 * 1024)], dtype=torch.int, device=torch.device('cuda'))
    if runner.world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()