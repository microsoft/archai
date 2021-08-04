#!/bin/bash
# fail if there are any errors
set -e
set -o xtrace

# requirement for Python 3.7 and below if pickle wasn't updated
# otherwise ray.init() fails if line below is not done
# conda install -y -c conda-forge pickle5

conda install -y -c anaconda pydot graphviz

bash scripts/apex_install.sh
pip install -e .