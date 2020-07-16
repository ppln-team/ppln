import albumentations as A
from albumentations import pytorch

from ppln.utils.misc import object_from_dict

from ..utils.registry import Registry

ALBUMENTATIONS = Registry("albumentations")


def make_albumentations(transforms, bbox_params=None):
    def build(config):
        if "transforms" in config:
            config["transforms"] = [build(transform) for transform in config["transforms"]]
        if hasattr(A, config["type"]):
            return object_from_dict(config, A)
        elif hasattr(pytorch, config["type"]):
            return object_from_dict(config, pytorch)
        else:
            return object_from_dict(config, ALBUMENTATIONS)

    return build({"type": "Compose", "transforms": transforms, "bbox_params": bbox_params})
