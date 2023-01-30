Package Structure
=================

The framework is composed of several packages, each with its own purpose and functionality. These packages are:

* API: This package provides a high-level API for building datasets and trainers. The API is designed to be user-friendly and provide abstractions that could be called during the NAS procedure.

* Common: This package contains various utility functions and classes that are used throughout Archai. These include functions for the configuration system, logging, and additional utilities.

* Datasets: This package contains datasets providers for use with Archai. These providers are commonly used in deep learning and include popular datasets like MNIST, CIFAR10, and ImageNet. Additionally, it supports loading datasets from external resources, such as Hugging Face's Hub and Torchvision.

* Discrete Search: This package provides tools for discrete hyperparameter search, including grid search and random search. This allows users to easily explore different hyperparameter configurations to find the best configuration for their model. In the context of NAS, this package also provides support for searching over discrete architectural choices, for example, the type of operations used in a model and the number of layers.

* ONNX: This package provides support for exporting models in the ONNX format, which can be used with other deep learning libraries and frameworks. This is useful for deploying models on devices that do not support PyTorch.

* Quantization: This package contains tools for quantizing deep learning models, which can be used to reduce the memory and computation requirements of a model. This is useful for deployment on devices with limited resources, like mobile phones or edge devices.

* Supergraph: This package provides support for defining and training models with dynamic architectures (graphs), where the architecture can change during the course of training.

* Trainers: This package contains training algorithms and utilities for training deep learning models.

In conclusion, Archai is designed to be a flexible and powerful NAS framework, with a focus on ease of use and abstracting away low-level details. The packages in Archai are designed to work together to provide a comprehensive solution for NAS, from defining candidate models to evaluating and deploying the best-performing model.
