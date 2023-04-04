# Readme

This folder contains some handy stuff for setting up an Azure account so you can run the code
in the [Azure](../azure/readme.md) folder.

First you will need to decide which Azure Subscription to use, install the
[Azure Command Line Interface](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
and run `az account set --subscription id` to make the this subscription your default.

## setup.ps1

This script creates the azure resources in your chosen subscription needed by the `runner.py`
script.  This includes a storage account for storing models and a status table, and a usage table.
The script contains a default `$plan_location` of `westus2`, feel free to change that to whatever
you need.  It also creates an azure docker container registry and AKS cluster, but using the
Kubernetes cluster for model quantization is optional, you can run the `runner.py` script without
AKS.

The setup script requires the following environment variables be set before hand:

- **SNPE_SDK** - points to a local zip file containing SNPE SDK version `snpe-1.64.0_3605.zip`
- **ANDROID_NDK** - points to a local zip file containing the Android NDK zip file version `android-ndk-r23b-linux.zip`
- **INPUT_TESTSET** - points to a local zip file containing 10,000 image test set from your dataset.

The [SNPE Readme](../snpe/readme.md) shows where to find those zip files.

After running this script you will see further instructions, first a docker command line in case you
want to build the docker image that runs in a kubernetes cluster.  Second, you will see a
`set MODEL_STORAGE_CONNECTION_STRING=...` that you can set for your system so that subsequent scripts
talk to the right azure storage account.

## Dockerfile

This builds a docker image that you can run in a Azure Kubernetes cluster that will do SNPE model
quantization in the cloud.  This frees up your Linux box that is managing Qualcomm devices and helps
you increase your Qualcomm device utilization.

The `Setup.ps1` script shows what docker commands to run to build the image, how to login to your
azure docker container registry, how to take your image for that container registry and push it
to Azure.  So you do not need to use the public docker.org container registry.  You will decide
what version number to attach to your image here and the same version needs to be specified in the
following `quant.yaml`.

## quant.yaml

Once the docker image is published you can configure your Azure Kubernetes cluster. First
you need to connect your local docker to this cloud service.  The Azure Portal has a connect
script to that under the AKS resource Overview there is a `Connect` button containing a string
like this:
```
az aks get-credentials --resource-group snpe-quantizaton-rg --name snpe-quantizer-aks
```
Run that locally and then you can push docker images to this registry.

Then you can use `kubectl apply -f quant.yaml` to configure the AKS custer.  Note that the version
of the image to use is specified in this file so you may need to edit the file and change the
version `1.13` to whatever you just tagged and pushed to the azure container registry.

Notice this yaml configures AKS to scale up to 100 nodes if necessary and the scaling is triggered
when a given node passes 40% CPU utilization.  You can tweak these numbers however you like to fit
your budget. But you may be surprised by how cheap AKS is. In a month of quantizing over 8000
models, my Azure cost analysis shows a total cost of around $8. A drop in the bucket compared to
model training costs. The AKS cluster auto-scales, so most of the time it is scaled down to 1 node
and sitting idle, generating very little cost.

You can run `kubectl get pods` to see what is running in Azure and you should see something like this:
```
NAME                              READY   STATUS              RESTARTS   AGE
snpe-quantizer-54dcf74c99-kfj8p   0/1     ContainerCreating   0          4s
snpe-quantizer-845c7cfcd8-q8kjh   1/1     Running             0          47h
```

## run.sh

This little script is used as the entry point to the Docker image, you will see this in the last
`RUN` command in the Dockerfile.
