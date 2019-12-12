from .priority import get_priority


class Hook(object):
    @property
    def priority(self):
        return get_priority('NORMAL')

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    @staticmethod
    def every_n_epochs(runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    @staticmethod
    def every_n_inner_iters(runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    @staticmethod
    def every_n_iters(runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    @staticmethod
    def end_of_epoch(runner):
        return runner.inner_iter + 1 == len(runner.data_loader)
