import logging
import tempfile

import pytest
import torch
import torch.nn as nn

from ppln.batch_processor import BaseBatchProcessor
from ppln.hooks import LogBufferHook, ProgressBarLoggerHook, TextLoggerHook, TimerHook
from ppln.runner import Runner


class SimpleBatchProcessor(BaseBatchProcessor):
    def train_step(self, model, batch, **kwargs):
        return {"loss": model(batch), "values": {}, "num_samples": len(batch)}

    def val_step(self, model, batch, **kwargs):
        return self.train_step(model, batch, **kwargs)

    def test_step(self, model, batch, **kwargs):
        pass


@pytest.fixture(scope="session")
def simple_runner():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = Runner(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        batch_processor=SimpleBatchProcessor(cfg=None),
        hooks=[TimerHook(), TextLoggerHook(), LogBufferHook(), ProgressBarLoggerHook(bar_width=10)],
        work_dir=tmp_dir,
    )

    return runner
