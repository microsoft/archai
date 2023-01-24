#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Fails if any errors
set -e
set -o xtrace

if python -c "import apex" &> /dev/null; then
    echo 'NVidia Apex is already installed'
else
    mkdir -p ~/GitHubSrc
    pushd ~/GitHubSrc
    rm -rf ./apex
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install --user -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
fi
