from ...utils.misc import master_only
from .. import BaseHook
from ..priority import Priority
from ..registry import HOOKS


@HOOKS.register_module
class BaseLoggerHook(BaseHook):
    @property
    def priority(self):
        return Priority.VERY_LOW

    @master_only
    def log(self, runner):
        raise NotImplementedError
