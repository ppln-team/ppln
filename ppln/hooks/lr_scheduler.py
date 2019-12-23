from math import cos, pi

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base import BaseHook
from .registry import HOOKS


class WarmupLrScheduler:
    def __init__(self, warmup=None, warmup_iters=0, warmup_ratio=0.1):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup)
                )
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

    def get_warmup_lr(self, cur_iters, regular_lr):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        else:
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in regular_lr]
        return warmup_lr


def _set_lr(optimizer, lr_groups):
    for param_group, lr in zip(optimizer.param_groups, lr_groups):
        param_group['lr'] = lr


@HOOKS.register_module
class LrSchedulerHook(BaseHook, WarmupLrScheduler):
    def __init__(self, name='base', by_epoch=True, warmup=None, warmup_iters=0, warmup_ratio=0.1, **kwargs):
        super().__init__(warmup=warmup, warmup_iters=warmup_iters, warmup_ratio=warmup_ratio)
        self.name = name
        self.by_epoch = by_epoch

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizers[self.name].param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in runner.optimizers[self.name].param_groups]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)
        _set_lr(runner.optimizers[self.name], self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                _set_lr(runner.optimizers[self.name], self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter, self.regular_lr)
                _set_lr(runner.optimizers[self.name], warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                _set_lr(runner.optimizers[self.name], self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter, self.regular_lr)
                _set_lr(runner.optimizers[self.name], warmup_lr)


@HOOKS.register_module
class ReduceLROnPlateauHook(BaseHook, WarmupLrScheduler):
    def __init__(self, name='base', metric_name='main_loss', warmup=None, warmup_iters=0, warmup_ratio=0.1, **kwargs):
        super().__init__(warmup=warmup, warmup_iters=warmup_iters, warmup_ratio=warmup_ratio)
        self.name = name
        self.metric_name = metric_name
        self.kwargs = kwargs
        self.scheduler = None
        self.regular_lr = []

    def before_run(self, runner):
        self.scheduler = ReduceLROnPlateau(optimizer=runner.optimizers[self.name], **self.kwargs)
        self.regular_lr = [group['lr'] for group in runner.optimizers[self.name].param_groups]

    def before_train_iter(self, runner):
        cur_iter = runner.iter

        if self.warmup is None or cur_iter > self.warmup_iters:
            return
        elif cur_iter == self.warmup_iters:
            _set_lr(runner.optimizers[self.name], self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter, self.regular_lr)
            _set_lr(runner.optimizers[self.name], warmup_lr)

    def after_val_epoch(self, runner):
        self.scheduler.step(runner.log_buffer.output[self.metric_name])


@HOOKS.register_module
class FixedLrSchedulerHook(LrSchedulerHook):
    def __init__(self, **kwargs):
        super(FixedLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


@HOOKS.register_module
class StepLrSchedulerHook(LrSchedulerHook):
    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


@HOOKS.register_module
class ExpLrSchedulerHook(LrSchedulerHook):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module
class PolyLrSchedulerHook(LrSchedulerHook):
    def __init__(self, power=1., **kwargs):
        self.power = power
        super(PolyLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return base_lr * (1 - progress / max_progress)**self.power


@HOOKS.register_module
class InvLrSchedulerHook(LrSchedulerHook):
    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


@HOOKS.register_module
class CosineLrSchedulerHook(LrSchedulerHook):
    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrSchedulerHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))
