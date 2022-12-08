30-Minute Tutorial
==================

To begin, ensure that Archai is properly :doc:`installed <../getting_started/installation>`. Then, familiarize yourself with the concept of Neural Architecture Search (NAS), as well as the specific algorithm being used, such as by reading the relevant papers or tutorials.

What is Archai?
---------------

Archai is an open-source platform for Neural Architecture Search (NAS) that offers a range of features to support efficient experimentation and reproducibility. It provides abstractions and unifications for different phases of NAS, and allows users to add new datasets by simply adding configuration files. Archai also provides common infrastructure for training and evaluation, and allows users to sweep over multiple macro parameters to generate the Pareto front. It offers support for both differentiable search and growing networks approaches, as well as efficient PyTorch code and structured logging.

Overview
--------

The process of using Archai is divided into two phases: search and evaluation. In the search phase, the algorithm uses a proxy dataset (a smaller dataset, such as CIFAR-10) to find the optimal network architecture. In the evaluation phase, this network is scaled up and trained on the target dataset (such as ImageNet) to generate the final trained PyTorch model weights.

.. mermaid::

    graph LR
        algo{Algorithm} --> mdesc[Model Description]
        mdesc --> eval[Evaluation]
        eval --> a0[Architecture 0]
        eval --> a1[Architecture 1]
        eval --> a2[Architecture 2]
        d(Dataset) --> algo
        sdesc(Search Description) --> algo
        hp(Hyperparameters) -.-> algo
        hp -.-> eval

        classDef bold stroke-width:4px;
        class algo bold

Configuration System
--------------------

Archai uses a sophisticated YAML based configuration system. As an example, you can view `configuration <https://github.com/microsoft/archai/blob/master/benchmarks/confs/algos/darts.yaml>`_ for running DARTS algorithm. At first it may be a bit overwhelming, but this ensures that all config parameters are isolated from the code and can be freely changed.

The config for search phase is located in ``nas/search`` section while for evaluation phase is located in ``nas/eval`` section. You will observe settings for data loading in ``loader`` section and training in ``trainer`` section. You can easily change the number of epochs, batch size, among others.

One great thing about Archai configuration system is that you can override any setting specified in YAML through command line as well. For instance, if you want to run evaluation only for 200 epochs instead of default 600, specify the path of the value in YAML separated by "." like this:

.. code-block:: bash

    python scripts/main.py --algos darts --nas.eval.trainer.epochs 200

Core Classes
------------

The core classes of Archai include:

* **ExperimentRunner**: This class is the entry point for running the algorithm through its ``run`` method. It has methods to specify what to use for search and evaluation that algorithm implementer can override;
* **Searcher**: This class enables the performance of search by simply calling its ``search`` method. Algorithm implementers should inherit from this class and override methods as needed;
* **Evaluater**: This class allows the evaluation of a given model by simply calling its ``evaluate`` method. Algorithm implementers should inherit from this class and override methods as needed;
* **Model**: This class is derived from PyTorch's ``nn.Module`` and adds additional functionality to represent architecture parameters;
* **ModelDesc**: This is a model description that describes the architecture of the model. It can be converted to a PyTorch model using the ``Model`` class anytime. It can be saved to YAML and loaded back. The purpose of the model description is to allow for a machine-readable data structure so that we can easily edit this model programmatically and scale it during the evaluation process;
* **ModelDescBuilder**: This class builds the ``ModelDesc`` that can be used by ``Searcher`` or evaluated by ``Evaluater``. Typically, algorithm implementers will inherit from this class to produce the model that can be used by the ``Searcher``;
* **ArchTrainer**: This class takes in an instance of ``Model`` and trains it using the specified configuration;
* **Finalizers**: This class takes a super network with learned architecture weights and uses a strategy to select edges to produce the final model;
* **Op**: This class is derived from ``nn.Module`` but has additional functionality to represent deep learning operations such as max pool or convolutions with architecture weights. It also can implement finalization strategies if NAS methods are using super networks for searching.

Running Archai
--------------

To run the main script using the command line, specify the ``--algos`` switch, followed by the desired algorithm. For example, to run the DARTS algorithm, you would use the following code:

.. code-block:: bash

    python scripts/main.py --algos darts

This will run the script in toy mode, using a reduced dataset and a limited number of epochs to quickly check that everything is working as expected. A full run, which uses the full dataset and a larger number of epochs, can take several days to complete on a single V100 GPU. To run a full run, add the ``--full`` switch:

.. code-block:: bash

    python scripts/main.py --algos darts --full

.. note::

    Archai uses CIFAR-10 as the default dataset.

When the script completes, you should see two directories in the ``~/logdir`` directory: one for search and one for evaluation. The search directory should contain a ``final_model_desc.yaml`` file, which contains the description of the network found by the search process, and a ``model.pt`` file, which is a trained PyTorch model generated by scaling the found architecture and training it for longer. You will also find a ``log.log`` file with human-readable logs, as well as a ``log.yaml`` file with a machine-readable version of the logs.

In addition to DARTS, Archai also supports other algorithms that you can use instead. You can specify multiple algorithms in a comma-separated list to run multiple algorithms at the same time.

.. tip::
    You can use Archai with Visual Studio Code debug mode: Ctrl+Shift+D.
