#!/bin/bash
source /home/archai/.profile
mkdir -p /home/archai/experiment

if [[ ! -d /home/archai/datasets/FaceSynthetics ]]; then
    mkdir -p /home/archai/datasets/FaceSynthetics
    pushd /home/archai/datasets/FaceSynthetics/
    azcopy copy https://nasfacemodels.blob.core.windows.net/downloads/099000.zip .
    unzip 099000.zip
    rm -rf 099000.zip
    popd
fi

pushd $SNPE_ROOT
source $SNPE_ROOT/bin/envsetup.sh -o /usr/local/lib/python3.6/dist-packages/onnx
popd
pushd /home/archai/experiment

while true
do
    python3 -u /home/archai/archai/devices/azure/runner.py
    if [ $? != 0 ]; then
      echo "Script returned an error code!"
    fi
    echo "Sleeping for 30 seconds..."
    sleep 30
done
