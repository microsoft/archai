#!/bin/bash
mkdir -p /home/archai/experiment

export INPUT_DATASET=/home/archai/datasets/FaceSynthetics
if [[ ! -d $INPUT_DATASET ]]; then
    mkdir -p $INPUT_DATASET
    pushd $INPUT_DATASET
    azcopy copy https://nasfacemodels.blob.core.windows.net/downloads/099000.zip .
    unzip 099000.zip
    rm -rf 099000.zip
    popd
fi

python -m olive.snpe.configure

pushd $SNPE_ROOT
source $SNPE_ROOT/bin/envsetup.sh -o $SNPE_ROOT/python36-env/lib/python3.6/site-packages/onnx
popd
pushd /home/archai/experiment

while true
do
    python -u /home/archai/image_segmentation/snpe/azure/runner.py
    if [ $? != 0 ]; then
      echo "Script returned an error code!"
    fi
    echo "Sleeping for 30 seconds..."
    sleep 30
done
