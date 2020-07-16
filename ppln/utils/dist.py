import functools
import os
import os.path as osp
import pickle
import shutil
import tempfile
from getpass import getuser
from socket import gethostname

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..fileio import io


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_host_info():
    return "{}@{}".format(getuser(), gethostname())


def init_dist(backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    torch.distributed.init_process_group(backend=backend, **kwargs)


def get_dist_info():
    initialized = is_dist_avail_and_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def all_gather_gpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def all_gather_cpu(data, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda")
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)
    # dump the part result to the dir
    io.dump(data, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    data_list = []
    for i in range(world_size):
        data = osp.join(tmpdir, f"part_{i}.pkl")
        data_list.append(io.load(data))

    return data_list
