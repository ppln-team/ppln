import albumentations as A

from ppln.utils.misc import object_from_dict


def build_albumentations(transforms):
    """
    Build transformation from albumentations library.
    Please, visit `https://albumentations.readthedocs.io` to get more information.

    Args:
        transforms (list): list of transformations to compose.
    """
    def build(config):
        if 'transforms' in config:
            config['transforms'] = [build(transform) for transform in config['transforms']]
        return object_from_dict(config, A)

    return build({
        'type': 'Compose',
        'transforms': transforms,
    })
