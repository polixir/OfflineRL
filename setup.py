#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("offlinerl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name='offlinerl',
    description="A Library for Offline RL(Batch RL)",
    url="https://agit.ai/Polixir/OfflineRL",
    version=get_version(),
    packages=find_packages(),
    author="SongyiGao",
    author_email="songyigao@gmail.com",
    python_requires=">=3.7",
    install_requires=[
        "aim==2.0.27",
        "fire",
        "loguru",
        "gym",
        "sklearn",
        "gtimer",
        "numpy",
        "ray",
    ],
    
)
