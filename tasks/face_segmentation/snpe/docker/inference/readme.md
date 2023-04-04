# Readme

This folder contains docker setup for creating a custom Azure ML docker container that can access
Qualcomm 888 dev kits via usb using the Android `adb` tool. This docker container is designed to run
in a local minikube cluster on the machine that has the Qualcomm 888 dev boards plugged in so that
these devices become available as an Azure ML Arc kubernetes compute cluster for use in [Azure ML
Pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk).
This way you can do a full Archai network search, Azure ML training of the models, SNPE
Quantization, and evaluation of inference times on the target Qualcomm 888 DSP hardware all in one
very cool Azure ML Pipeline.

First you will need to decide which Azure Subscription to use, install the
[Azure Command Line Interface](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
and run `az account set --subscription id` to make the this subscription your default.

The setup script requires the following environment variables be set before hand:

- **SNPE_SDK** - points to a local zip file containing SNPE SDK version `snpe-1.64.0_3605.zip`
- **ANDROID_NDK** - points to a local zip file containing the Android NDK zip file version `android-ndk-r23b-linux.zip`
- **INPUT_TESTSET** - points to a local zip file containing 10,000 image test set from your dataset.

The [SNPE Readme](../snpe/readme.md) shows where to find those zip files.

After running this script you will see further instructions, first a docker command line in case you
want to build the docker image that runs in a kubernetes cluster.

## Dockerfile

This builds a docker image that you can run in a Azure Kubernetes cluster that will do SNPE model
quantization in the cloud.  This frees up your Linux box that is managing Qualcomm devices and helps
you increase your Qualcomm device utilization.

The `Setup.ps1` script shows what docker commands to run to build the image, how to login to your
azure docker container registry, how to take your image for that container registry and push it
to Azure.  So you do not need to use the public docker.org container registry.  You will decide
what version number to attach to your image here and the same version needs to be specified in the
following `quant.yaml`.

## minikube

Once the docker image is published you can configure your local Kubernetes cluster. If you are
using `minikube` you can run this command to switch your `kubectl` command to operate on
minikube's docker environment: `eval $(minikube -p minikube docker-env)`.
