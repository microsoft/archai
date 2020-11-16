#!/bin/bash
#fail if any errors
set -e
set -o xtrace

if python -c "import apex" &> /dev/null; then
    echo 'NVidia Apex is already installed'
else
    mkdir -p ~/GitHubSrc
    pushd ~/GitHubSrc
    rm -rf ./apex # for some reason this exist in amlk8s
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
fi

if python -c "import nvidia.dali.pipeline" &> /dev/null; then
    echo 'NVidia dali is already installed'
else
    pip install --user nvidia-pyindex
    pip install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda100
fi

