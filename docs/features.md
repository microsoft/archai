# Archai Features

Archai is designed to unify several latest algorithms for Network Architecture Search into a common codebase allowing for much more generality as well as reproducibility with fair comparison. While enabling next generation research in NAS, we also aim to provide high quality turn-key implementations to rapidly try out these algorithms on your custom datasets and scenarios. This page describes several high level features available in Archai.

## Features

* **Declarative Approach and Reproducibility**: Archai incorporates yaml based config system with an additional ability to inherit from base yaml and share settings without needing to copy/paste. This design allows to easily mix and match parts of different algorithms without having to write code. The resulting yaml becomes self-documenting list of all hyper-parameters as well as bag-of-tricks enabled for the experiment. The goal is to make critical decisions explicit that otherwise may remain buried in the code making it harder to reproduce experiment for others. It is then also possible to perform several different experiments  by merely changing config that otherwise might need significant code changes.

* **Plug-n-Play Datasets**: Archai provides infrastructure so a new dataset can be added by simply adding new config. This allows for much faster experimentation for real-world datasets and leverage latest NAS research to benefit actual products.

* **Fair Comparison**: A recent crisis in the field of NAS is ability to fairly compare different techniques with *same* bag-of-tricks such that often makes more difference than NAS technique itself. Archai stipulates all the bag-of-tricks to be config driven and hence enforces ability to run different algorithms on a fair leveled field. Further, instead of different algorithms using vastly different training and evaluation code, Archai provides common infrastructure to again allow fair comparison.

* **Unifications and Abstractions**: Archai is designed to provide abstractions for various phases of NAS including architecture trainers, finalizers, evaluators and so on. Also the architecture is represented by model description expressed as yaml that can be "compiled" into PyTorch network. Unlike "genotypes" used in traditional algorithms, these model descriptions are far more powerful, flexible and extensible. This allows unifying several different NAS techniques including macro-search into a single framework. The modular architecture allows for extensions and modifications in just few lines of code.

* **Exhaustive Pareto Front Generation**: Archai allows to sweep over several macro parameters such as number of cells, number of nodes etc along with reports on model statistics such as number of parameters, flops, inference latency and model memory utilization to allow identify optimal model for desired constraints.

* **Differentiable Search vs Growing Networks**: Archai offers unified codebase for two mainstream approaches for NAS: differentiable search and growing networks one iteration at time (also called forward search). For growing networks, Archai also supports initializing weights from previous iteration for faster training.

* **Clean Codebase with Aggregated Best Practices**: Archai has leveraged several different popular codebases to extract different best practices in one codebase with an extensible and modular Pythonic design. Our hope is that Archai can serve as great starting point for future NAS research.

* **NAS for Non-Experts**: Archai enables quick plug-n-play for custom datasets and ability to run sweep of standard algorithms. Our goal is to present Archai as turn-key NAS solution for the non-expert.

* **Efficient PyTorch Code**: Archai implements best practices for PyTorch as well as implements efficient versions of algorithms such as bi-level optimizers that runs as much as 2X faster than original implementation.

* **Structured Logging**: Archai logs all information of the run in machine-readable structured yaml file as well as human readable text file. This allows to extract minute details of the run for comparisons and analysis.

* **Metrics and Reporting**: Archai collects all metrics including timings into machine readable yaml that can easily be analyzed. One can also run multiple combinations of parameters, each with multiple seeds and then compute mean as well as standard deviations over multiple seed tuns. Archai includes reporting component to generate metrics plots with standard deviation envelops and other details.

* **General Purpose Trainer**: Archai includes general purpose trainer that can be used to train any PyTorch model including handcrafted models. This trainer includes several best practices and along with all other infrastructure it would be useful to anyone even if NAS is not a primary focus. This trainer also supports features including support for multiple optimizers, warm up schedule, chunking support etc.

* **Mixed Precision and Distributed Runs**: Archai supports easy config driven distributed multi-GPU runs with or without mixed precision support that can make runs 2X-4X faster for TensorCore based GPUs such as NVidia V100s. Archai includes several best practices through NVidia Apex library as well as its own components such as distributed stratified sampler.

* **Development Mode**: The so called "toy" mode allows for quick end-to-end testing during development so you can develop on usual laptop and do a full runs in cloud. Archai also supports TensorWatch, TensorBoard and other debugging aids such as network visualization, timing logs etc.

* **Enhanced Archai Model**: The Archai `Model` class derives from PyTorch `nn.Module` but adds on features such as clear separation of architecture and non-architecture differentiable parameters.

* **Cross Platform**: Archai runs on Linux as well as Windows however distributed runs are currently only supported on Linux.
