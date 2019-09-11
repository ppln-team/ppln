import sys

from ppln.utils.timer import Timer


class ProgressBar(object):
    """A progress bar which can print the progress"""
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.timer = None
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    @staticmethod
    def _get_max_bar_width():
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                f'terminal width is too small ({terminal_width}), please consider '
                'widen the terminal for better progressbar '
                'visualization'
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if not self.task_num > 0:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.timer = Timer()

    def update(self, text=''):
        self.completed += 1
        elapsed = self.timer.since_start()
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
            sys.stdout.write(
                f'\r{text} '
                f'[{bar_chars}] '
                f'{self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, '
                f'elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s'
            )
        else:
            sys.stdout.write(
                f'{text} ', f'completed: {self.completed}, '
                f'elapsed: {int(elapsed + 0.5)}s, '
                f'{fps:.1f} tasks/s'
            )
        sys.stdout.flush()
