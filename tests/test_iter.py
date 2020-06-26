import shutil

import torch
from torch.utils.data import DataLoader

from ppln.runner import Runner


def test_iter(simple_runner: Runner):
    loader = DataLoader(torch.ones((10, 2)), batch_size=1)
    simple_runner.run({"train": loader, "val": loader}, max_epochs=2)
    import os

    print(f"{simple_runner.work_dir}/{os.listdir(simple_runner.work_dir)[0]}")
    # shutil.rmtree(simple_runner.work_dir)
