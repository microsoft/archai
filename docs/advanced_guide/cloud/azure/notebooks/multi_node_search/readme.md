# Multi-node Search

This sample shows how to use Azure ML to do partial training on a GPU cluster during an Archai
`EvolutionParetoSearch`.  Implementing the `AsyncModelEvaluator` interface an
`AmlTrainingValAccuracy` evaluator is able to do partial training of many models in parallel across
a GPU cluster using [Azure ML
pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk)
where the result is the best possible model model for `MNIST` classification.  The same technique
works for any task, we only use MNIST here to minimize the compute cost of the GPU cluster.

The [multi_node_search](multi_node_search.ipynb) notebook uses your Azure storage account and your
Azure ML workspace.  The Azure storage account will accumulate results while the search is in
progress.

The notebook first kicks off the master search pipeline which will look like this:

![search](images/search_pipeline.png)

The data prep component simply copies the MNIST dataset to a pipeline blob store for use as input
not only on this search job but also in all the partial training jobs to come.

The `EvolutionParetoSearch` will do 5 iterations, and during that time it will kick off one new AML
pipeline for each iteration that will train the batch of models that the `AmlTrainingValAccuracy`
needs to get validation accuracy numbers for.  You can get the log output of the search job by
looking at the "Outputs + logs" in the Azure Machine Learning Studio.

For example, the following partial training pipeline was created to do partial training of 36 models in a GPU cluster:

![training.png](images/training.png)

Azure ML automatically keeps the GPU cluster busy feeding all the models until they are all trained.
In this case we create an 8 node GPU cluster, and so you will see the jobs completing in batches of 8.

You can see all the pipelines created dynamically by the search using your Azure ML Pipelines
dashboard. Below you see 5 partial training iterations followed by a final full training job and
since we have an 8-gpu cluster doing the work the entire job finished in 1 hour 14 minutes. It did
partial training on a total of 65 models, and full training on 32.

![dashboard](images/dashboard.png)

You can also run the cell titled `Plots` in the notebook multiple times to watch how the pareto curves are
shaping up, you will see something like this after 5 iterations have completed.

![pareto](images/epochs1.png)

When the search is finished the next cell in the notebook will download the results including a file named 'models.json' that reports the best models found by the search:


```json
{
  "init_num_models": 10,
  "partial_training_epochs": 1,
  "full_training_epochs": 10,
  "id_bc52a6ee_4ef2_4327_8c8c_3bba2165c3ea": {
    "archid": "(1, 3, 32)",
    "val_acc": 0.3953
  },
  "id_1250cb3f_0fc9_4cad_a4f2_e627db5c66e8": {
    "archid": "(1, 1, 32)",
    "val_acc": 0.2237
  }
}
```

You can run the last cell of the notebook to compare the inference accuracy of the top
model using the ONNX runtime locally on the resulting downloaded .onnx model.

You should see something like this:

```
Inference pass rate is  99.57 %.
How does this compare with the training validation accuracy of 0.9957
```

## Design

So how does all this work.  The overall architecture looks like this:

![arch](images/workflow.png)

Where the main search pipeline launches [search.py](scripts/search.py) script on a cheap CPU virtual
machine since it doesn't need much horse power.  This search pipeline starts with a quick data-prep
step to ensure our dataset is in a shared Azure blob store, then it does the Archai search.  When
the search is complete we have a `pareto.json` file containing the id's of the best models, and we
then kick off a full training run for those models using a child pipeline.

This `search.py` script plugs the [aml_training_evaluator.py](scripts/aml_training_evaluator.py)
into the `EvolutionParetoSearch` objectives, and the `AmlTrainingValAccuracy` is an
`AsyncModelEvaluator` meaning that the search algorithm calls `send` to pass all the models in the
current iteration then it calls `fetch_all` to get the results.

So it is the `fetch_all` method that creates a new AML pipeline for that iteration,
dynamically adding a partial training job for each model, each one of those commands
invokes [train.py](scripts/train.py).  `train.py` uses [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to train the model, which means the model is a `LightningModule`.

But how does each partial training job send results back to the master search pipeline
you might ask? Great question, I'm glad you asked. The training jobs are given access
to the azure storage account where there is a `status` table that they write to and
a `models` blob store they can write the trainged .onnx files to.

You can look at this status table to get a great idea of what is going on:
![status](images/status.png)

Here you can see some models are still `training` and some are completed: `trained`.
You can also see what the model architecture is for each one. When a job is finished
it will have a published `val_acc` column containing the validation accuracy reported
by the partial training.

The `AmlTrainingValAccuracy` evaluator collects these numbers and returns them to the
search algorithm which will use them to select the best models to iterate on.

## Scripts

The code behind this notebook is organized into the following folders:

### data_prep

`prep_data_store.py` this script contains the code that runs in the data prep Azure ML pipeline component.
It's job in life is to move the training dataset from the MnistDatasetProvider into the cloud using
the mounted Azure blob store location given as a command line argument.  Azure ML makes this look
like a local folder, but in fact, it is automatically copied to an azure blob store by Azure ML because
the path has `mode="rw_mount"` in the pipeline component definition.  This is a cool feature of Azure ML
that makes it super simple to test all these scripts locally, but then they all "just work" when
running in the cloud.

This script is in a different folder from the other scripts because this way ensures maximum reuse
of the output dataset during the development of your other training script. Often times those need
more debugging and this will save on cloud compute by maximizing the reuse of this node in each
submitted Azure ML pipeline

### scripts

`aml_training_evaluator.py` provides the class `AmlTrainingValAccuracy` which implements the Archai
`AsyncModelEvaluator` interface.  This is the core of this entire tutorial, showing how to use the
`AsyncModelEvaluator` interface to get parallel training happening on an Azure ML GPU cluster.

`commands.py` provides some of the Azure ML pipeline `command`s used in the
[multi_node_search](multi_node_search.ipynb) notebook, moved here just so the notebook doesn't
become overly verbose.

`mnist_data_module.py` provides a Pytorch `LightningDataModule` wrapper on the `MnistDatasetProvider`
so we can use the Pytorch Lightning `Trainer`.

`model.py` this is our configurable pytorch CNN model that is also a `LightningModule`. This class builds
different model architectures during the Archai search

`monitor.py` this provides a `JobCompletionMonitor` class that monitors the status of training jobs in
our Azure Storage Table returning all the validation accuracies when the jobs are complete.
This is used by the `AmlTrainingValAccuracy` model evaluator.

`search.py` this is the main Archai search component that will run in our Azure ML pipeline.  It
uses the `AmlTrainingValAccuracy` model evaluator as a search objective among others.  It is also
designed so it can run locally for debugging, with the training happening in the cloud.

`train.py` this is the model training script that uses the Pytorch Lightning `Trainer` to train a
specific model architecture.  This will be used across all the nodes of your GPU cluster to train as
many jobs in parallel as you want.  It reports status to the Azure Storage Table so that the
`JobCompletionMonitor` knows when they finish and what the validation accuracy was.

`training_pipeline.py` this script can start an entirely new Azure ML Pipeline for doing some
parallel training of models, where each training component uses `train.py`.  This same script is
used to kick off partial training from the `AmlTrainingValAccuracy` and the final full training run
performed in the main pipeline defined in the [multi_node_search](multi_node_search.ipynb) notebook.

`utils.py` a few small helper functions used by the notebook, moved here just to reduce the notebook
size.
