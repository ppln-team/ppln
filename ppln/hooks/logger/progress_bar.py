import sys
from colorama import Fore, Style
from ..hook import Hook
from ...utils.misc import master_only
from ...utils.progress_bar import ProgressBar


class ProgressBarLoggerHook(Hook):
    def __init__(self):
        super(ProgressBarLoggerHook, self).__init__()
        self.bar = None

    def before_epoch(self, runner):
        self.bar = ProgressBar(task_num=len(runner.data_loader))

    @master_only
    def after_epoch(self, runner):
        sys.stdout.write(f'\n')

    def after_train_iter(self, runner):
        self.log(runner)

    def after_val_iter(self, runner):
        self.log(runner)

    @master_only
    def log(self, runner):
        epoch_color = Fore.YELLOW
        mode_color = (Fore.RED, Fore.BLUE)[runner.train_mode]
        text_color = (Fore.CYAN, Fore.GREEN)[runner.train_mode]
        epoch_text = f'{epoch_color}epoch:{Style.RESET_ALL} {runner.epoch + 1:<4}'
        log_items = [
            (' ' * 11, epoch_text)[runner.train_mode], f'{mode_color}{runner.mode:<5}{Style.RESET_ALL}',
            f'{text_color}lr:{Style.RESET_ALL} {", ".join([f"{lr:.3e}" for lr in runner.lr])}'
        ]
        for name, value in runner.log_buffer.output.items():
            if isinstance(value, float):
                value = f'{value:.4f}'
            log_items.append(f'{text_color}{name}:{Style.RESET_ALL} {value}')
        self.bar.update(f"{' | '.join(log_items)}")
