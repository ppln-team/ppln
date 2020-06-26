from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class BatchProcessorOutput:
    loss: torch.Tensor
    num_samples: int
    values: Dict[str, float]
    target: torch.Tensor = None
    prediction: torch.Tensor = None
    fields: Dict[str, Any] = field(default_factory=dict)


class BaseBatchProcessor(ABC):
    @abstractmethod
    def train_step(self, model, batch, **kwargs) -> BatchProcessorOutput:
        raise NotImplementedError
