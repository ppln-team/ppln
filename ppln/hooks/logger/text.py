import datetime
from collections import OrderedDict

from ...utils.misc import master_only
from ..registry import HOOKS
from .base import BaseLoggerHook
from .utils import get_lr


@HOOKS.register_module
class TextLoggerHook(BaseLoggerHook):
    def __init__(self):
        self.time_sec_tot = 0
        self.start_iter = None
        self.json_log_path = None

    def before_run(self, runner):
        self.start_iter = runner.iter

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            lr_str = ''.join(
                [f"{name}_lr: {', '.join([f'{lr:.3e}' for lr in lrs])}, " for name, lrs in log_dict['lr'].items()]
            )
            log_str = f'Epoch [{log_dict["epoch"]}][{log_dict["iter"]}/{len(runner.data_loader)}]\t{lr_str}'
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * len(runner.data_loader))
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, data_time: {log_dict["data_time"]:.3f}, '
        else:
            log_str = f'Epoch({log_dict["mode"]}) [{log_dict["epoch"]}][{log_dict["iter"]}]\t'
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in ['mode', 'Epoch', 'iter', 'lr', 'time', 'data_time', 'epoch']:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    @master_only
    def log(self, runner):
        log_dict = OrderedDict()
        mode = runner.mode
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        log_dict['lr'] = get_lr(runner.optimizers)
        if mode == 'train':
            log_dict['time'] = runner.log_buffer.output['time']
            log_dict['data_time'] = runner.log_buffer.output['data_time']
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, runner)

    def after_epoch(self, runner):
        self.log(runner)
