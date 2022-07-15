#!/bin/bash

command="~/git/archai/devices/azure/loop.sh"
next=1

DeviceRunner () {
    folder="~/snpe/experiment$next"
    echo "Setting up screen session for device $1 in $folder ..."
    mkdir -p $folder
    cd $folder
    screen -dmS "$1" "$command" "--device $1 --no_quantization"
    next=$((next+1))
}

for usb in $(adb devices | awk '$2 ~ /device/ {print $1}')
do
    DeviceRunner $usb
done

screen -ls