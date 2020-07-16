import functools
import pydoc
import time


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def object_from_dict(d, parent=None, **default_kwargs):
    assert isinstance(d, dict) and "type" in d
    kwargs = d.copy()
    object_type = kwargs.pop("type")

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, dict) and "type" in value:
            value = object_from_dict(value)
            kwargs[key] = value

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.
    """

    def __init__(self, func):
        functools.update_wrapper(wrapper=self, wrapped=func)
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
