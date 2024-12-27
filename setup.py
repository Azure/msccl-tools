# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name='msccl',
    version='2.3.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'msccl = msccl.__main__:main',
        ],
    },
    scripts = [],
    install_requires=[
    ],
    python_requires='>=3.8',
)
