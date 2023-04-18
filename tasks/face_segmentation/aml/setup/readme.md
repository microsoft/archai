# Setup

This script contains the session setup used on the machine that is connected
to the Qualcomm boards for running N screen sessions that run the loop.sh
script for each Qualcomm board.

If you want to also cleanup stale kubernetes pods, you can add `--cleanup_stale_pods`
once you have configured `az login` and `az aks get-credentials --resource-group $resource_group --name $aks_cluster `
so that the runner script can call `cleanup_stale_pods.py`.