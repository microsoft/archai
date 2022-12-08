30-Minute Tutorial
==================

If you would like to follow through this tutorial, please make sure you have [installed](../getting-started/install.md) Archai.

What is Neural Architecture Search?
-----------------------------------

Network architecture search is the process of finding best performing neural network architectures for a given problem. Usually, we have a dataset such as [ImageNet](https://www.image-net.org/) and we want to figure out how to arrange operations such as convolutions and pooling to create a neural network that has the best classification accuracy. Many practical situations also require that the architecture must fit into device memory or should use only certain number of flops. NAS shines in such problems because finding optimal/near-optimal architecture requires a lot of guess work and human effort.

So how do we automatically find a good neural architecture? There are a number of algorithms proposed by the NAS research community in the past few years. In this tutorial we will learn how to use existing algorithms in Archai. We will also get the overview of how Archai works and finally we will implement a popular NAS algorithm called DARTS to show how you can implement your own algorithm (often only with a few lines of code). If you are not familiar with DARTS, we recommend [reading the paper](https://arxiv.org/abs/1806.09055) or [basic overview](https://towardsdatascience.com/intuitive-explanation-of-differentiable-architecture-search-darts-692bdadcc69c) first and then coming back here.

What is Archai?
---------------

Archai is a powerful, flexible codebase for network architecture search (NAS). It unifies several latest algorithms for NAS into a common codebase, allowing for greater generality and reproducibility with fair comparison.

Overview
--------

Now that we are familiar with how to run NAS algorithms, use command line switches and configuration system, let's briefly review how Archai works.

Some of the most popular NAS methods like DARTS use what is known as a *proxy dataset*. So if you want to find network that works on ImageNet, you may use smaller dataset such as CIFAR10 (or a subset of ImageNet) for search. After you have found the network for CIFAR10, you scale it up, change model stems and train for longer epochs. This final network now can be used for ImageNet. So these type of NAS methods works in two phases: search and evaluation.

Archai uses YAML based model description that can be "compiled" to PyTorch model at anytime. The advantage of using such description is that model description can easily be generated, modified, scaled and visualized. Before the search starts, Archai creates the model description that represents the super network that algorithms like DARTS can use for differentiable search. The output of the search process is again model description for the best model that was found. This output model description is than used by evaluation phase to scale it up and generate the final trained PyTorch model weights. This process is depicted in below diagram:

![Run DARTS in VSCode](../assets/img/archai_workflow.png)

We will now see how this workflow can be implemented in Archai.

Key Features
------------

**Declarative Approach and Reproducibility**: Archai uses a YAML-based configuration system with the ability to inherit from base YAML files, allowing users to easily mix and match different algorithms and settings without having to write code. The resulting YAML file serves as a self-documenting list of all hyperparameters and bag-of-tricks used in the experiment. This design makes critical decisions explicit and helps ensure reproducibility for other users.

**Plug-n-Play Datasets**: Archai provides infrastructure that allows users to add new datasets simply by adding new configuration files, enabling faster experimentation on real-world datasets and leveraging the latest NAS research.

**Fair Comparison**: Archai stipulates that all bag-of-tricks be config-driven, ensuring that different algorithms can be compared on a level playing field. Additionally, Archai provides common infrastructure for training and evaluation, allowing for fair comparison between algorithms.

**Unifications and Abstractions**: Archai provides abstractions for various phases of NAS, including architecture trainers, finalizers, evaluators, and more. The architecture is represented by a model description expressed in YAML that can be "compiled" into a PyTorch network. These model descriptions are far more powerful, flexible, and extensible than traditional "genotypes", allowing for the unification of several different NAS techniques into a single framework.

**Exhaustive Pareto Front Generation**: Archai allows users to sweep over multiple macro parameters, such as the number of cells and nodes, and reports on model statistics such as the number of parameters, FLOPS, inference latency, and model memory utilization, enabling users to identify optimal models for their desired constraints.

**Differentiable Search vs Growing Networks**: Archai offers a unified codebase for two mainstream approaches to NAS**: differentiable search and growing networks one iteration at a time (also known as forward search). For growing networks, Archai also supports initializing weights from previous iterations for faster training.

**Clean Codebase with Aggregated Best Practices**: Archai has leveraged several popular codebases to extract best practices and integrate them into a single, extensible, and modular Pythonic design. It is our hope that Archai can serve as a great starting point for future NAS research.

**NAS for Non-Experts**: Archai enables quick plug-n-play for custom datasets and allows users to run a sweep of standard algorithms. We have designed Archai to be a turn-key NAS solution for non-experts.

**Efficient PyTorch Code**: Archai implements best practices for PyTorch and provides efficient implementations of algorithms such as bi-level optimizers that run up to 2X faster than the original implementations.

**Structured Logging**: Archai logs all information from the run in a machine-readable structured YAML file as well as a human-readable text file. This allows for easy analysis of experiments and facilitates collaboration between researchers.

**Structured logging**: Archai logs all run information in a machine-readable structured yaml file as well as a human-readable text file. This allows for detailed analysis and comparison of runs.

**Metrics and reporting**: Archai collects all metrics, including timings, in a machine-readable yaml file that can be easily analyzed. Multiple combinations of parameters can be run with multiple seeds, and the results can be averaged and plotted with standard deviation envelops.

**General purpose trainer**: Archai includes a general-purpose trainer that can be used to train any PyTorch model, including handcrafted models. This trainer includes best practices and is useful even if NAS is not the primary focus. The trainer also supports features such as multiple optimizers, warm up schedules, and chunking support.

**Mixed precision and distributed runs**: Archai supports config-driven distributed multi-GPU runs with or without mixed precision support, which can make runs up to 4 times faster on TensorCore-based GPUs such as NVidia V100s. Archai includes best practices from the NVidia Apex library as well as its own components, such as the distributed stratified sampler.

**Development mode**: The "toy" mode allows for quick end-to-end testing during development, so you can develop on a laptop and run full experiments in the cloud. Archai also supports tools such as TensorWatch, TensorBoard, and network visualization for debugging.

**Enhanced Archai Model**: The Archai Model class derives from PyTorch's nn.Module but adds features such as clear separation of architecture and non-architecture differentiable parameters.

**Cross-platform compatibility**: Archai runs on Linux and Windows, although distributed runs are currently only supported on Linux.

Configuration System
--------------------

Archai uses a sophisticated YAML based configuration system. As an example, you can [view configuration](https://github.com/microsoft/archai/blob/master/benchmarks/confs/algos/darts.yaml) for running DARTS algorithm. At first it may be a bit overwhelming, but this ensures that all config parameters are isolated from the code and can be freely changed. The config for search phase is located in `nas/search` section while for evaluation phase is located in `nas/eval` section. You will observe settings for data loading in `loader` section and training in `trainer` section. You can easily change the number of epochs, batch size etc.

One great thing about Archai config system is that you can override any setting specified in YAML through command line as well. For instance, if you want to run evaluation only for 200 epochs instead of default 600, specify the path of the value in YAML separated by `.` like this:

```bash
python scripts/main.py --algos darts --nas.eval.trainer.epochs 200
```

You can read in more detail about features available in Archai config system later.

Core Classes
------------

At the heart of Archai are the following classes:

* **ExperimentRunner**: This class is the entry point for running the algorithm through its `run` method. It has methods to specify what to use for search and evaluation that algorithm implementer can override.
* **Searcher**: This class allows to perform search by simply calling its `search` method. Algorithm implementer should inherit from this class and override methods as needed.
* **Evaluater**: This class allows to perform evaluation of given model by simply calling its `evaluate` method. Algorithm implementer should inherit from this class and override methods as needed.
* **Model**: This class is derived from PyTorch ``nn.Module` and adds additional functionality to represent architecture parameters.
* **ModelDesc**: This is model description that describes the architecture of the model. It can be converted to PyTorch model using the `Model` class anytime. It can be saved to YAML and loaded back. The purpose of model description is to simply allow machine readable data structure so we can easily edit this model programmatically and scale it during the evaluation process.
* **ModelDescBuilder**: This class builds the `ModelDesc` that can be used by `Searcher` or evaluated by `Evaluater`. Typically, algorithm implementer will inherit from this class to produce the model that can be used by the `Searcher`.
* **ArchTrainer**: This class takes in the instance of `Model` and trains it using the specified configuration.
* **Finalizers**: This class takes a super network with learned architecture weights and uses strategy to select edges to produce the final model.
* **Op**: This class is derived from `nn.Module` but has additional functionality to represent deep learning operations such as max pool or convolutions with *architecture weights*. It also can implement finalization strategy if NAS method is using super networks for searching.

Running Archai
--------------

Running NAS algorithms built into Archai is easy. You can use either command line or Visual Studio Code. Using command line, run the main script specifying the `--algos` switch:

```bash
python scripts/main.py --algos darts
```

Notice that the run completes within a minute or so. This is because we are using reduced dataset and epochs just to quickly see if everything is fine. We call this *toy mode*. Doing a full run can take couple of days on single V100 GPU. To do a full run, just add the `--full` switch:

```bash
python scripts/main.py --algos darts --full
```

When you run these algorithms, Archai used cifar10 dataset as default. Later we will see how you can use other datasets or even bring our own custom dataset easily.

By default, Archai produces output in `~/logdir` directory. You should see two directories: One for search and other for evaluation often refered colloqually in docs as `eval`. Evaluation in NAS means taking architecture/s that were found during the search phase and training them from scratch on the full dataset for longer and often with lots of enhancements added on. The search folder should have `final_model_desc.yaml` which contains the description of network that was found by DARTS. You will find `model.pt` which is trained PyTorch model generated after scaling the architecture found by search process and training it for longer. You should also see `log.log` which captures the human readable logs and `log.yaml` that is machine readable version of logs.

You can also use other algorithms [available](algorithms.md) in Archai instead of DARTS. You can also run multiple algorithms by specifying them in comma separated list.

We will use [Visual Studio Code](https://code.visualstudio.com/) in this tutorial however you can also use any other editor. Archai comes with preconfigured run configurations for VS Code. You can run DARTS in debug mode by opening the archai folder and then choosing the Run button (Ctrl+Shift+D):

![Run DARTS in VSCode](../assets/img/vscode_run_darts.png)
