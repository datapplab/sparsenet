##!/usr/bin/env python


from __future__ import absolute_import
from setuptools import setup, find_packages

VERSION = '0.0.1'

requirements = [
    'tensorflow>=2.0.0',
    'numpy>=1.17.0'
]


setup(
    # Metadata
    name='sparsenet',
    version=VERSION,
    author='Weijun Luo',
    author_email='luo_weijun@yahoo.com',
    url='https://github.com/datapplab/sparsenet',
    description='Implementation of sparse neural networks',
    # long_description=readme,
    # long_description_content_type='text/markdown',
    license='GPLv3',

    # Package info
    packages=find_packages(),

    #
    zip_safe=True,
    install_requires=requirements,
    python_requires='>=3.6',

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',    
    ]
)