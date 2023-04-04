#!/bin/bash

MODEL_NAME="model"

if [ "$1" == "--help" ] ; then
    echo "### Usage: convert_onnx.sh [model_name]"
    echo "Converts the given onnx model to .dlc then quantizes it."
    echo "Default model path is 'model/model.onnx'."
    exit 1
fi 

if [ "$1" != "" ]; then
   MODEL_NAME=$1
fi   

if [ ! -f "model/${MODEL_NAME}.onnx" ]; then
    echo "### Model does not exist: model/${MODEL_NAME}.onnx"
    exit 1
fi

mkdir -p ./snpe_models

# pb 2 dlc
snpe-onnx-to-dlc \
    -i "model/${MODEL_NAME}.onnx" \
    -d input_0 "1,3,256,256" \
    --input_layout input_0 NCHW \
    --out_node "output_0" \
    -o "snpe_models/${MODEL_NAME}.dlc" \
    #--debug

# quantize
snpe-dlc-quantize \
    --input_dlc "snpe_models/${MODEL_NAME}.dlc" \
    --input_list "data/quant/input_list.txt" \
    --output_dlc "snpe_models/${MODEL_NAME}.quant.dlc" \
    --use_enhanced_quantizer
