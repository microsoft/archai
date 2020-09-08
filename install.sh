#!/bin/bash
#fail if any errors
set -e
set -o xtrace

# ray.init() fails if below is not done
conda install -y -c conda-forge pickle5
conda install -y -c anaconda pydot graphviz

bash scripts/apex_install.sh
pip install -e .