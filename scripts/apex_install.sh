#!/bin/bash
#fail if any errors
set -e
set -o xtrace

if python -c "import apex" &> /dev/null; then
    echo 'NVidia Apex is already installed'
else
    mkdir -p ~/GitHubSrc
    pushd ~/GitHubSrc
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
fi

if python -c "import nvidia.dali.pipeline" &> /dev/null; then
    echo 'NVidia dali is already installed'
else
    pip install nvidia-pyindex
    pip install nvidia-dali-cuda100
fi

