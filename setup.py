#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='taputapu',
      version='0.0.1',
      license='GPL',
      author='Sofia Ares Oliveira',
      description='Tools for image processing',
      install_requires=[
          'numpy',
          'pandas',
          'imageio',
          'tqdm',
          'pillow'
      ],
      packages=find_packages(where='.'),
      zip_safe=False)
