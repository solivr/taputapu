#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='im_tools',
      version='0.0.1',
      license='GPL',
      author='Sofia Ares Oliveira',
      description='Tools for image processing',
      install_requires=[
          'numpy',
          'pandas',
          'imageio',
          'tqdm'
      ],
      packages=find_packages(where='.'),
      zip_safe=False)
