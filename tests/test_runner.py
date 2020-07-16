import os.path as osp

import pytest
import torch
from torch.utils.data import DataLoader

from ppln.hooks import CheckpointHook, EarlyStoppingHook
from ppln.runner import Runner


@pytest.mark.parametrize("num_samples", [1, 3])
@pytest.mark.parametrize("max_epochs", [1, 3])
def test_iter(simple_runner: Runner, num_samples, max_epochs):
    loader = DataLoader(torch.ones(num_samples, 2), batch_size=1)
    simple_runner.run({"train": loader, "val": loader}, max_epochs=max_epochs)
    assert simple_runner.epoch == max_epochs - 1
    assert simple_runner.iter == simple_runner.max_iters
    assert simple_runner.inner_iter == num_samples - 1
    assert simple_runner.mode == "val"


def test_save_checkpoint(simple_runner):
    loader = DataLoader(torch.ones((3, 2)), batch_size=1)
    simple_runner.add_hook(CheckpointHook())
    simple_runner.run({"train": loader, "val": loader}, max_epochs=1)
    best_path = osp.join(simple_runner.work_dir, "best.pth")
    epoch1_path = osp.join(simple_runner.work_dir, "epoch_1.pth")

    assert osp.exists(best_path)
    assert osp.exists(epoch1_path)
    assert osp.realpath(best_path) == osp.realpath(epoch1_path)
    torch.load(best_path)


def test_early_stopping(simple_runner):
    patience = 2
    loader = DataLoader(torch.ones((3, 2)), batch_size=1)
    simple_runner.add_hook(EarlyStoppingHook(patience=patience))
    simple_runner.run({"train": loader, "val": loader}, max_epochs=10)
    assert simple_runner.epoch == patience
