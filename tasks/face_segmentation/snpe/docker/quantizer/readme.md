# Readme

This folder contains some handy stuff for setting up an Azure account so you can run the code in the
[Azure](../../azure/readme.md) folder and create a docker image for running SNPE model quantization
jobs on a kubernetes cluster. You can also run this docker image in a Linux container on Windows
using the Docker Desktop for Windows.

First you will need to decide which Azure Subscription to use, install the
[Azure Command Line Interface](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
and run `az account set --subscription id` to make the this subscription your default.

## setup.ps1

This [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux?view=powershell-7.3)
script creates the azure resources in your chosen subscription needed by the `runner.py` script.
This includes a storage account for storing models and a status table, and a usage table. The script
contains a default `$plan_location` of `westus2`, feel free to change that to whatever you need.  It
also creates an azure docker container registry and AKS kubernetes cluster, but using the Kubernetes
cluster for model quantization is optional, you can run the `runner.py` script without AKS.

The setup script requires the following environment variables be set before hand:

- **SNPE_SDK** - points to a local zip file containing SNPE SDK (we have tested version `snpe-2.5.0.4052.zip`)
- **ANDROID_NDK** - points to a local zip file containing the Android NDK zip file (we have tested version `android-ndk-r25c-linux.zip`)
- **INPUT_TESTSET** - points to a local zip file containing 10,000 image test set from your dataset.

The [SNPE Readme](../../snpe/readme.md) shows where to find those zip files.

After running this script you will see further instructions, first a docker command line in case you
want to build the docker image that runs in a kubernetes cluster.  Second, you will see a
`set MODEL_STORAGE_CONNECTION_STRING=...` that you can set for your system so that subsequent scripts
talk to the right azure storage account.  On linux this would be an export in your `~/.profile`, and
don't forget the double quotes.

```
export MODEL_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=mymodels;AccountKey=...==;EndpointSuffix=core.windows.net"
```

## Dockerfile

This builds a docker image that you can run in a Azure Kubernetes cluster that will do SNPE model
quantization in the cloud.  This frees up your Linux box that is managing Qualcomm devices and helps
you increase your Qualcomm device utilization.

The `setup.ps1` script shows what docker commands to run to build the image, how to login to your
azure docker container registry, how to take your image for that container registry and push it
to Azure.  So you do not need to use the public docker.org container registry.  You will decide
what version number to attach to your image here and the same version needs to be specified in the
following `quantizer.yaml`.

You can also test your docker image locally by running:
```
docker run -e MODEL_STORAGE_CONNECTION_STRING=$MODEL_STORAGE_CONNECTION_STRING -it <image_id>
```

If you need to debug your docker image interactively you can run this instead:
```
docker run -it <image_id> /bin/bash
```
Then you can poke around the `run.sh` and other things to verify things manually.

## Publish image to your Azure Container Registry

First you need to tag your newly created image with the correct name:
```
docker tag <image_id> snpecontainerregistry001.azurecr.io/quantizer:1.27
```

You can find the correct version in the `quantizer.yaml` file that was updated by `setup.ps1`.

Then you can push this image to your Azure Container Registry named `snpecontainerregistry001`. You
can configure your local docker so it can talk to this Azure Kubernetes cluster (AKS). The Azure
Portal has a connect script under the AKS resource Overview. You will see a `Connect` button
that shows a string like this:
```
az aks get-credentials --resource-group snpe-quantizaton-rg --name snpe-quantizer-aks
```
Run that locally and then you can push docker images to this registry.:

```
docker push snpecontainerregistry001.azurecr.io/quantizer:1.27
```

Again, make sure you specify the right version here.  The `setup.ps1` script will automatically
increment this version number each time it runs in case you need to push new versions of this image.

## quantizer.yaml

Then you can use `kubectl apply -f quantizer.yaml` to configure the AKS custer.  Note that the version
of the image to use is specified in this file so you may need to edit the file and change the
version `1.13` to whatever you just tagged and pushed to the azure container registry.

Notice this yaml configures AKS to scale up to 100 nodes if necessary and the scaling is triggered
when a given node passes 40% CPU utilization.  You can tweak these numbers however you like to fit
your budget. But you may be surprised by how cheap AKS is. In a month of quantizing over 8000
models, the Azure cost analysis shows a total cost of around $8. A drop in the bucket compared to
model training costs. The AKS cluster auto-scales, so most of the time it is scaled down to 1 node
and sitting idle, generating very little cost.

This quantizer runs in the `snpe` kubernetes namespace, and you can make this your default namespace
by running:
```
kubectl config set-context --current --namespace=snpe
```

You can run `kubectl get pods` to see what is running in Azure and you should see something like this:
```
NAME                              READY   STATUS              RESTARTS   AGE
snpe-quantizer-54dcf74c99-kfj8p   0/1     ContainerCreating   0          4s
snpe-quantizer-845c7cfcd8-q8kjh   1/1     Running             0          47h
```

You can watch what these pods are doing by running:
```
kubectl logs snpe-quantizer-54dcf74c99-kfj8p -f
```
And you will see some output like this:
```
Sleeping for 30 seconds...
Using storage account: "nasfacemodels"
snpe-quantizer-d9f4b6c96-jsb7q: model test is running on: clovett-14_e6dc0375
# skip entity test because someone else is working on it
No work found.
Sleeping for 30 seconds...
```
This is good and means the pod is waiting for work to show up in the `status` table in your
Azure storage account. You can now use the [upload.py](../../azure/upload.py) script to upload a
face segmentation ONNX model to do a test run.  You can train one of these models using
[train.py](../../../train.py) to train one of good model architectures listed in
[archs/snp_target](../../../archs/snp_target).

## run.sh

This little script is used as the entry point to the Docker image, you will see this in the last
`RUN` command in the Dockerfile.
