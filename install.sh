#!/bin/bash
#fail if any errors
set -e
set -o xtrace

# ray.init() fails if below is not done
conda install -c conda-forge pickle5

bash tools/apex_install.sh
pip install -e .