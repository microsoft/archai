#!/bin/bash


MODEL_NAME="model"

if [ "$1" == "--help" ] ; then
    echo "### Usage: convert_tf.sh [model_name]"
    echo "Converts the given tensorflow model to .dlc then quantizes it."
    echo "Default model path is 'model/model.pb'."
    exit 1
fi 

if [ "$1" != "" ]; then
   MODEL_NAME=$1
fi   

if [ ! -f "model/${MODEL_NAME}.pb" ]; then
    echo "### Model does not exist: model/${MODEL_NAME}.pb"
    exit 1
fi

mkdir -p ./snpe_models

# pb 2 dlc
snpe-tensorflow-to-dlc \
    -i "model/${MODEL_NAME}.pb" \
    -d input_rgb "1,256,256,3" \
    --out_node "logits_cls" \
    -o "snpe_models/${MODEL_NAME}.dlc" \
    --show_unconsumed_nodes
    #--debug

# quantize
snpe-dlc-quantize \
    --input_dlc "snpe_models/${MODEL_NAME}.dlc" \
    --input_list "data/quant/input_list.txt" \
    --output_dlc "snpe_models/${MODEL_NAME}.quant.dlc" \
    --use_enhanced_quantizer
