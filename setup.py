import codecs
import os
import re

from setuptools import find_packages, setup


def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)


def get_contents(*args):
    """Get the contents of a file relative to the source distribution directory."""
    with codecs.open(get_absolute_path(*args), 'r', 'UTF-8') as handle:
        return handle.read()


def get_version(*args):
    """Extract the version number from a Python module."""
    contents = get_contents(*args)
    metadata = dict(re.findall('__([a-z]+)__ = [\'"]([^\'"]+)', contents))
    return metadata['version']


setup(
    name='ppln',
    version=get_version('ppln', '__init__.py'),
    author='Miras Amir',
    author_email='amirassov@gmail.com',
    description='PyTorch runner',
    long_description_content_type='text/markdown',
    url='https://github.com/ppln-team/ppln',
    packages=find_packages(),
    install_requires=[
        'torchvision', 'torch>=1.1', 'tqdm', 'PyYAML', 'colorama', 'addict', 'jpeg4py', 'opencv-python',
        'albumentations'
    ],
    setup_requires=['pytest-runner'],
    python_requires='>=3.6.0'
)
