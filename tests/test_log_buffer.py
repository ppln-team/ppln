import os

import pytest
import torch.distributed as dist
from torch.multiprocessing import Process

from ppln.utils.log_buffer import LogBuffer


def run(rank, world_size):
    count = 5
    log_buffer = LogBuffer()
    log_buffer.update({"value": rank}, count=count)
    assert len(log_buffer.n_history["value"]) == 1
    assert len(log_buffer.value_history["value"]) == 1
    assert log_buffer.n_history["value"][0] == count
    assert log_buffer.value_history["value"][0] == rank
    log_buffer.synchronize_between_processes()
    outputs = log_buffer.average()
    assert len(log_buffer.n_history["value"]) == world_size
    assert len(log_buffer.value_history["value"]) == world_size
    assert log_buffer.n_history["value"] == [count for _ in range(world_size)]
    assert log_buffer.value_history["value"] == [i for i in range(world_size)]
    assert outputs["value"] == sum([i * count for i in range(world_size)]) / sum([count for _ in range(world_size)])


def init_process(rank, size, fn, backend="nccl"):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


@pytest.mark.parametrize("world_size", [1, 2])
def test_log_buffer(world_size):
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
