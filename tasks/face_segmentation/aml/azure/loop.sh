#!/bin/bash
# If the python app runs out of memory due to various leaks in python libraries
# the process terminates with 'killed', this loop will restart the runner.
script_dir="$(dirname ${BASH_SOURCE})"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate snap37

while true
do
    python ${script_dir}/runner.py $@
    if [ $? != 0 ]; then
      exit 0
    fi
    echo "sleeping for 30 seconds..."
    sleep 30
done
