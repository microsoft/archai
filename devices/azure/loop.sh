#!/bin/bash
# If the python app runs out of memory due to various leaks in python libraries
# the process terminates with 'killed', this loop will restart the runner.
script_dir="$(dirname ${BASH_SOURCE})"
while true
do
    python ${script_dir}/runner.py $@
    if [ $? != 0 ]; then
      exit 0
    fi
    echo "sleeping for 30 seconds..."
    sleep 30
done
