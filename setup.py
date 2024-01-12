#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='madronarlenvs',
      version='0.0.1',
      description='MadronaRLEnvs',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'torch',
          'numpy',
          'tqdm',
          'overcooked-ai==1.1.0',
          'gym==0.22.0',
          'tensorboard',
          'packaging',
          'scipy'
      ],
      )
