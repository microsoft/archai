# Azure Setup

This Azure code connects to an Azure Storage account by a connection string setup in an environment variable named
`MODEL_STORAGE_CONNECTION_STRING`.

There is a handy [setup.ps1](../docker/quantizer/setup.ps1) powershell script that will setup a new Azure Storage
account and print this connection string key for you using the Azure CLI.

You can get the connection string from your Azure Storage account under `Access Keys` and `Show Keys` and copy the one
named `Connection string`.

In Linux you should use double quotes around the connection string like this:

```
export MODEL_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=mymodels;AccountKey=...==;EndpointSuffix=core.windows.net"
```

You'll use it a lot so it is handy if you put it in your `~/.profile`.

Then you can use the scripts here as follows:

1. `upload.py` - upload a new model, it will allocate a friendly name for your model if you
can't think of one.  All the charts and graphs and tables are more readable if you stick with
the allocated friendly names.  But the actual model file names can be whatever large name you need.

1. `download.py` - can download all azure blob assets associated with the given friendly name.
This can include the .onnx model, all test results, and converted .dlc models.

1. `priority_queue.py` - is just a helper class.

1. `reset.py` - sometimes you want to re-test a model if something went wrong, this can reset
the `state` of a job by it's friendly name.

1. `status.py` - can download all the status info from the Azure Table in .csv format.

1. `delete.py` - sometimes a model turns out to be junk, so you can delete a row in the table and it's
associated azure blob folder with this handy script.

1. `runner.py` - this is the magic script that runs everything.  See below.

## Runner

Then to get the ball rolling create a temp folder and run this:

```
mkdir -p ~/experiment
python ~/git/archai/tasks/face_Segmentation/aml/azure/runner.py --working ~/experiment
```

This will monitor the Azure blob store for new work to do, and run those jobs in priority order.  If you also provide a
`--device` option pointing to the `adb device` for a Qualcomm 888 Dev Board then it will also run the quantized models
on that device and report the performance and F1 score results.

If you setup a quantization only runner in the cloud using the `docker/quantizer` image, you can pass
`--no_quantization` argument when you have a `--device` so that the local runs do not do quantization. This will stop
your linux machine from getting overloaded with quantization work so it can focus on the SNPE device workloads.