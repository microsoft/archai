#!/bin/bash
#fail if any errors
set -e

nvidia-smi --list-gpus

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "*****************************************"
echo "Using $gpu_count GPUS"
echo "*****************************************"

set -e -o xtrace


python -m torch.distributed.launch --nproc_per_node=$gpu_count scripts/main.py $*