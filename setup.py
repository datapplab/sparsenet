#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.0.1'

requirements = [
    'tensorflow>=2.0.0',
    'numpy>=1.17.0'
]


setup(
    # Metadata
    name='sparse',
    version=VERSION,
    author='Weijun Luo',
    author_email='luo_weijun@yahoo.com',
    url='https://github.com/datapplab/sparse',
    description='Implementation of sparse neural networks',
    # long_description=readme,
    # long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    # packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,
	python_requires='>=3.6',

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',    
    ],
)