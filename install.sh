#!/usr/bin/env bash
# For setting up a virtual environment
# This assumes that Conda or Miniconda has been installed
# And a virtual environment created and activated

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# for unpacking the Potsdam 1_DSM.rar file
conda install conda-forge::rarfile

#export CUDA_HOME=???  Do we need this?

pip install -U openmim && mim install mmcv-full=="1.7.1"
pip install mmsegmentation==0.30.0

# expected formatter
pip install yapf==0.40.1

# Install the project as a module using setup.py
pip install -v -e .
# ^^^ If using pip >= 25.0 this will need to be changed. Message:
# DEPRECATION: Legacy editable install of aerialseg==0.0.1 from
# file:///home/davidbeers/dev/AerialFormer (setup.py develop) is
# deprecated. pip 25.0 will enforce this behaviour change. A
# possible replacement is to add a pyproject.toml or enable
# --use-pep517, and use setuptools >= 64. If the resulting
# installation is not behaving as expected, try using
# --config-settings editable_mode=compat. Please consult the
# setuptools documentation for more information. Discussion
# can be found at https://github.com/pypa/pip/issues/11457