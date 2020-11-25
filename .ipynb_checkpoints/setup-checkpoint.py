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
        "gym",
        "gtimer",
        "numpy",
        #"git+https://gitee.com/geak/dm_control.git#egg=dm_control",
        #"git+https://gitee.com/geak/mjrl.git#egg=mjrl",
        #"git+https://gitee.com/geak/d4rl.git#egg=d4rl",
        #"git+https://gitee.com/geak/tianshou.git#egg=tianshou",
    ],
    
)