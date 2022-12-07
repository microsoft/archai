Key Features
============

Archai is a powerful, flexible codebase for network architecture search (NAS). It unifies several latest algorithms for NAS into a common codebase, allowing for greater generality and reproducibility with fair comparison.

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