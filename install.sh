#!/bin/bash
#fail if any errors
set -e
set -o xtrace

bash tools/apex_install.sh
pip install -e .