#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Runs an interactive bash within the container
# Enhanced security by gVisor / without GPUs
docker run --rm \
    --runtime=runsc \
    --name nvidia22.10-archai0.7.0 \
    --shm-size=10g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NCCL_P2P_LEVEL=NVL \
    -it nvidia22.10-archai0.7.0:latest