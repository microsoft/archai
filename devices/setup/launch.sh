#!/bin/bash

command="~/git/archai/devices/azure/loop.sh"
next=1

DeviceRunner () {
    folder="~/snpe/experiment$next"
    mkdir -p $folder
    echo cd $folder
    echo screen -dmS $1 $command --device $1 --no_quantization
    next=$((next+1))
}

echo "# run the following lines to setup the screen sessions..."

for usb in $(adb devices | awk '$2 ~ /device/ {print $1}')
do
    DeviceRunner $usb
done
