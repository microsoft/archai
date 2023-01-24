#!/bin/bash


# Fails if any errors
set -e
set -o xtrace

# Below will be required for Python 3.7 and below if pickle wasn't updated
# otherwise ray.init() fails if below is not done
# conda install -y -c conda-forge pickle5

conda install -y -c anaconda pydot graphviz
bash apex_install.sh
pip install -e .[dev]