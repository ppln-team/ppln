import os.path as osp

import torch
from torch.utils.data import DataLoader

from ppln.hooks import CheckpointHook, EarlyStoppingHook
from ppln.runner import Runner


def test_iter(simple_runner: Runner):
    max_epochs = 3
    loader = DataLoader(torch.ones((10, 2)), batch_size=1)
    simple_runner.run({"train": loader, "val": loader}, max_epochs=max_epochs)
    assert simple_runner.epoch == max_epochs - 1
    assert simple_runner.iter == max_epochs * len(loader)
    assert simple_runner.inner_iter == len(loader) - 1
    assert simple_runner.max_iters == max_epochs * len(loader)
    assert simple_runner.mode == "val"


def test_save_checkpoint(simple_runner):
    loader = DataLoader(torch.ones((10, 2)), batch_size=1)
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
    loader = DataLoader(torch.ones((10, 2)), batch_size=1)
    simple_runner.add_hook(EarlyStoppingHook(patience=patience))
    simple_runner.run({"train": loader, "val": loader}, max_epochs=10)
    assert simple_runner.epoch == patience
