import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ppln',
    version='0.2',
    author='Miras Amir',
    author_email='amirassov@gmail.com',
    description='Universal PyTorch runner',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/amirassov/youtrain',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
