#!/bin/bash
source /home/chris/.profile
mkdir -p /home/chris/experiment

if [[ ! -d /home/chris/datasets/FaceSynthetics ]]; then
    mkdir -p /home/chris/datasets/FaceSynthetics
    pushd /home/chris/datasets/FaceSynthetics/
    azcopy copy https://nasmodelstorage.blob.core.windows.net/downloads/099000.zip .
    unzip 099000.zip
    rm -rf 099000.zip
    popd
fi

pushd $SNPE_ROOT
source $SNPE_ROOT/bin/envsetup.sh -o /usr/local/lib/python3.6/dist-packages/onnx
popd
pushd /home/chris/experiment

while true
do
    python3 -u /home/chris/snpe_runner/azure/runner.py
    if [ $? != 0 ]; then
      echo "Script returned an error code!"
    fi
    echo "Sleeping for 30 seconds..."
    sleep 30
done
