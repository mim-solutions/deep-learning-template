#!/bin/bash

set -e #exit immediately after one of the commands failed

pip install --upgrade pip

pip install robbytorch \
            hydra-core \
            pytorch-lightning \
            albumentations \
            numpy \
            pandas \
            scikit-image \
            matplotlib \
            pyinstrument \
            jupyter \
            tqdm \
            wandb \
            toml \
            types-toml typing-extensions mypy \
            autopep8 flake8 pycodestyle \