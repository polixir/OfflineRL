#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("batchrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name='batchrl',
    description="A Library for Batch RL(Offline RL)",
    url="https://agit.ai/Yi/batchrl.git",
    version=get_version(),
    author="SongyiGao",
    author_email="songyigao@gmail.com",
    python_requires=">=3.7",
    install_requires=[
        "aim==2.0.27",
        "fire",
        "loguru",
        "gym",
        "gtimer",
        "numpy",
        "tianshou",
    ],
    
)
