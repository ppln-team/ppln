from typing import List, NoReturn, Optional

from ..utils.freeze import freeze_modules, lock_norm_modules
from .base import BaseClosureHook
from .priority import Priority
from .registry import HOOKS


@HOOKS.register_module
class ModelFreezeHook(BaseClosureHook):
    def __init__(self, modules: List[str], train: bool = False, unfreeze_epoch: Optional[int] = None) -> NoReturn:
        super().__init__(freeze_modules, root_modules=modules)
        self.train = train
        self._unfreeze_epoch = float("inf") if unfreeze_epoch is None else unfreeze_epoch

    @property
    def priority(self) -> Priority:
        return Priority.HIGHEST  # Must be higher than DDPHook and NormalizationLockHook

    def before_run(self, runner) -> NoReturn:
        self._func(model=runner.model, requires_grad=False, train=self.train)
        runner.logger.info("Layers are frozen")

    def before_train_epoch(self, runner) -> NoReturn:
        unfreeze_epoch = self._unfreeze_epoch - 1

        if runner.epoch < unfreeze_epoch:
            self._func(model=runner.model, requires_grad=False, train=self.train)
            runner.logger.info("Layers are re-frozen")

        elif runner.epoch == unfreeze_epoch:
            self._func(model=runner.model, requires_grad=True, train=True)
            runner.logger.info("Layers are de-frozen")


@HOOKS.register_module
class NormalizationLockHook(BaseClosureHook):
    def __init__(self, train: bool = False, requires_grad: Optional[bool] = True) -> NoReturn:
        super().__init__(lock_norm_modules, train=train, requires_grad=requires_grad)

    @property
    def priority(self) -> Priority:
        return Priority.VERY_HIGH  # Must be higher than ddp wrapper

    def before_run(self, runner) -> NoReturn:
        self._func(model=runner.model)

    def before_train_epoch(self, runner) -> NoReturn:
        self._func(model=runner.model)
