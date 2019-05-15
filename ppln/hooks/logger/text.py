import datetime
from collections import OrderedDict

import torch

from .utils import get_max_memory
from .. import Hook


class TextLoggerHook(Hook):
    def __init__(self):
        self.time_sec_tot = 0
        self.start_iter = None
        self.json_log_path = None

    def before_run(self, runner):
        self.start_iter = runner.iter

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            lr_str = ', '.join([f'{lr:.5f}' for lr in log_dict['lr']])
            log_str = f'Epoch [{log_dict["epoch"]}][{log_dict["iter"]}/{len(runner.data_loader)}]\tlr: {lr_str}, '
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * len(runner.data_loader))
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, data_time: {log_dict["data_time"]:.3f}, '
                log_str += f'memory: {log_dict["memory"]}, '
        else:
            log_str = f'Epoch({log_dict["mode"]}) [{log_dict["epoch"] - 1}][{log_dict["iter"]}]\t'
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in ['mode', 'Epoch', 'iter', 'lr', 'time', 'data_time', 'memory', 'epoch']:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        runner.logger.debug(log_str)

    def log(self, runner):
        log_dict = OrderedDict()
        # training mode if the output contains the key "time"
        mode = runner.mode
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        log_dict['lr'] = [lr for lr in runner.lr]
        if mode == 'train':
            log_dict['time'] = runner.log_buffer.output['time']
            log_dict['data_time'] = runner.log_buffer.output['data_time']
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = get_max_memory(runner)
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, runner)

    def after_epoch(self, runner):
        self.log(runner)
