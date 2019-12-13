from . import hooks as all_hooks
from .hooks import BaseHook
from .utils.misc import object_from_dict


class HookList:
    def __init__(self, hooks):
        self.hooks = []
        for hook in hooks:
            self.add(hook)

    def __call__(self, runner, action):
        for hook in self.hooks:
            getattr(hook, action)(runner)

    def add(self, hook):
        """Add a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
        """
        if isinstance(hook, dict):
            hook = object_from_dict(hook, all_hooks)
        elif not isinstance(hook, BaseHook):
            raise TypeError

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self.hooks) - 1, -1, -1):
            if hook.priority >= self.hooks[i].priority:
                self.hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self.hooks.insert(0, hook)
