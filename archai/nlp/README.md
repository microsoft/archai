# Natural Language Processing with Archai

Natural Language Processing (NLP) models use hardware advancements to solve more complex tasks. Nevertheless, such advancements also lead to an increased number of parameters, raising concerns regarding production-ready environments and low-resource devices.

Archai provides a straightforward alternative to find more efficient models through Neural Architecture Search (NAS), furnishing an ideal place to prototype and implement searches of autoregressive transformer-based architectures. Essentially, the idea is to keep everything simple while offering developers and researchers every single tool to fulfill their needs.

Use NLP with Archai if you need a package or wish to:

* ⚡ Fast-experiment transformer-based architectures;
* 📂 Design or use pre-loaded language modeling tasks;
* 📈 Increase your efficiency without losing effectiveness;
* 🔬 Find new architectures under certain constraints.

## Getting started: 60 seconds with Archai-NLP

Installation of the bleeding-edge version is easy as pie. Please clone this repository and run the following command line:

```bash
pip install -e .
```

After installing all the requirements, one can train a default model (NVIDIA's Transformer-XL) with just a single command line, as follows:

```bash
python archai/nlp/train.py
```

Finally, with another single command line, one can extract the Pareto front of the default search (also with NVIDIA's Transfomer-XL), as follows:

```bash
python archai/nlp/search.py
```

## Table of contents

 * [Data Loading and Utilities](#data-loading-and-utilities)
    * [Corpus](#corpus)
    * [Vocabularies and Tokenizers](#vocabularies-and-tokenizers)
    * [Iterators](#iterators)
 * [Transformer-based Architectures](#transformer-based-architectures)
    * [Available Architectures](#available-architectures)
    * [Adding New Architectures](#adding-new-architectures)
    * [Training a Model](#training-a-model)
 * [Neural Architecture Search](#neural-architecture-search)
    * [Evolutionary Search](#evolutionary-search)
    * [Finding the Groud-Truth](#finding-the-ground-truth)
    * [Extracting the Pareto Frontier](#extracting-the-pareto-frontier)
    * [Comparing the Frontiers](#comparing-the-frontiers)
 * [Architecture Compression](#architecture-compression)
    * [ONNX](#onnx)
    * [Quantization](#quantization)
 * [Metrics Scoring](#metrics-scoring)
 * [Support](#support)
    * [Contributions](#contributions)
    * [Team](#team)
    * [License](#license)
    * [Trademark](#trademark)

## Data Loading and Utilities

In a Natural Language Processing task, the first step is to encode raw pieces of information (e.g., text) into more appropriate structures, such as numerical vectors/tensors. Essentially, the general data loading pipeline is implemented by the `datasets` package performed as follows:

📄 **Data**: acquisition, cleaning and pre-processing;

📰 **Corpus**: creation;

📑 **Vocabulary/Tokenizer**: creation and tokenizer training;

🔖 **Iterator**: task-related iterating, such as causal language modeling.

### Corpus

As aforementioned, the corpus stands for a collection of pre-processed texts, which will be converted/trained into vocabularies and tokenizers. Although it is straightforward to add a new corpus, Archai implements the following ones out-of-the-box in the `datasets/corpus` module:

* [WikiText-2 and WikiText-103](https://arxiv.org/abs/1609.07843);
* [Penn Treebank (ptb)](https://catalog.ldc.upenn.edu/LDC99T42);
* [1 Billion Word (lm1b)](http://www.statmt.org/lm-benchmark);
* [English Wikipedia (enwik8 and text8)](http://mattmahoney.net/dc/textdata.html).

### Vocabularies and Tokenizers

After defining the loaded data (dataset type), one can produce a new vocabulary and train a new tokenizer. There are several vocabulary/tokenization methods, depending on the task that will be employed or the model that will be used. Thus, Archai implements in the `datasets/tokenizer_utils` package the following vocabularies and tokenizers:

* [Word-based](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/tokenizer_utils/word_vocab.py);
* [Byte-Level Byte-Pair Encoding (BBPE)](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/tokenizer_utils/bbpe_vocab.py);
* [GPT-2](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/tokenizer_utils/gpt2_vocab.py).

### Iterators

Finally, the last step of creating datasets and data loaders is to provide a task-compliant iterator, i.e., a data loader that produces samples and labels according to the desired task.

Note that every implemented iterator is available at the `datasets/lm_iterators` module, while their corresponding tasks are briefly described in the following sections.

#### Causal Language Modeling

The Causal Language Modeling (CLM) task aims to predict a `t+1` token based on the previous `t` tokens, i.e., predict a token followed by a sequence of tokens. In such a task, the model only attends to the left part of the context (previously seen tokens), which is often the setting used by auto-regressive and text-generation models. Archai implements three types of CLM iterators, as follows:

* [Ordered iterator](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/lm_iterators.py#L7);
* [Shuffled iterator](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/lm_iterators.py#L97);
* [Multi-file iterator](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/lm_iterators.py#L177).

#### Masked Language Modeling

*Currently, this type of iterator is not supported by Archai.*

## Transformer-based Architectures

Transformers have become one of the most employed architectures throughout the last years, mainly due to their handling of sequential data. Their architecture is often structured in layers composed of self-attention mechanisms, which capture and weigh the significance of each part of the data throughout the timesteps.

Although Archai has been built to be independent of models and architectures, i.e., fosters every type of neural architectures search, we opted to implement a `models` package and provide a few state-of-the-art architectures and samples that can be used out-of-the-box.

### Available Architectures

Archai provides a lazy-loading system that loads desired classes on-demand, which removes the user from the burden of knowing how every aspect of the library works. Thus, with such a system in hands, one can only care about the [model's dictionary](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/common/model_dict.py) and provide the correct classes to be loaded.

#### NVIDIA's Memory Transformer

A reasonably new architecture that can deal with more extended context has been proposed by [Dai et al.](https://arxiv.org/abs/1901.02860), denoted as Transformer-XL. Such an architecture has been re-implemented by NVIDIA with optimization and faster training time in mind, being baptized as [NVIDIA's Memory Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL/pytorch).

*This architecture is implemented by Archai within the `mem_transformer` reference and is available at the `models/mem_transformer` package.*

#### Huggingface's GPT-2

One of the most well-known transformer-based architectures is the Generative Pre-Trained Transformer (GPT), widely implemented in software, applications, and natural language systems. Currently, we only support [Huggingface's GPT-2](https://github.com/huggingface/transformers/tree/master/src/transformers/models/gpt2) implementation.

*This architecture is implemented by Archai within the `hf_gpt2` reference and is available at the `models/hf_gpt2` package.*

#### Huggingface's Transformer-XL

Archai also supports [Huggingface's Transformer-XL](https://github.com/huggingface/transformers/tree/master/src/transformers/models/transfo_xl) implementation, which is also derived from NVIDIA's code, with slight differences, such as learnable embedding parameters per layer instead of per model. Also, such a code is compliant with Huggingface's Transformers package, though not as fast as NVIDIA's Memory Transformer.

*This architecture is implemented by Archai within the `hf_transfo_xl` reference and is available at the `models/hf_transfo_xl` package.*

### Adding New Architectures

Archai is independent of architectures and can be virtually used within every model type, as long as they follow some guidelines.

Such guidelines are depicted by the [`ArchaiModel`](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/models/model_base.py) class, which directly inherits from the `torch.nn.Module` class. Essentially, every new implemented architecture should inherit from `ArchaiModel` and be added to the [`ModelDict`](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/common/model_dict.py#L23), which stands for the available models that can be loaded within the lazy loader.

Briefly speaking, these are the steps to implement a new architecture:

1. Create a new folder with the model's identifier inside the `models` package, for example, `transformer`;
2. Inside the created folder, create a `model_transformer.py` to hold the model's architecture and a `config_transformer.py` if the model should be available with ONNX exports;
3. Adds the corresponding implemented classes to the `ModelDict` under an uppercased string key that reflects the model's identifier, e.g., `TRANSFORMER`. The key values should come in a tuple format and follow the types defined by the [`ClassType`](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/common/model_dict.py#L12), i.e., `MODEL`, `ONNX_CONFIG` and `ONNX_MODEL`.
4. Finally, the new model can be directly used within the training script, as long as it is available within the `--model` flag.

### Training a Model

One can produce a new model by simply training an architecture with the proposed pipeline. Archai is responsible for saving every needed output, such as training/evaluation logs, checkpoints, and configuration files.

```bash
python archai/nlp/train.py --help 
```

## Neural Architecture Search

One of the foremost goals of Archai is to perform efficient Neural Architecture Searches and find optimal architectures that meet some desired guidelines. Thus, we offer a `nas` module that implements some customizable pipelines that enable users to find their most suitable architectures given a set of constraints.

The Transformer-based NAS pipeline is organized as follows:

* Conduct the evolutionary search to find suitable samples (architecture configurations);
* Submit ground-truth jobs for the entire population that has been found during the search;
* Extract a proxy Pareto frontier from all samples seen during the evolutionary search;
* Find candidate points in the proxy Pareto frontier and submit them for full training;
* Compare between ground-truth and proxy Pareto frontiers.

### Evolutionary Search

The whole NAS idea is structured as an evolutionary search for transformer-based architectures, where users can define the parameters to be searched and also their constraints that should be met.

The first step is to conduct the search and find a set of Pareto points that meet the constraints, as follows:

```bash
python archai/nlp/search.py --phase run_search --help 
```

Essentially, the search will find the best configuration file (which can be used to create the model) for the desired architecture under the input constraints. Traditionally, our search considers the number of non-embedding parameters and the model's latency.

### Finding the Ground-Truth

After finding possible configurations through the evolutionary search, we can submit some ground-truth jobs for further comparison:

```bash
python archai/nlp/search.py --phase submit_gt_jobs --help 
```

*Such a step is valid to determine whether proxy points (without training) and close or distant from the ground-truth points (with training).*

### Extracting the Pareto Frontier

Alternatively, our `search.py` script allows users in extracting ground-truth Pareto frontier from all samples seen during the evolutionary search, which can be invoked as follows:

```bash
python archai/nlp/search.py --phase extract_pareto --help 
```

Further, it is also possible to match the proxy Pareto frontier points (found in the previous step) with the baseline and submit the selected points on the Pareto frontier for full training, as follows:

```bash
python archai/nlp/search.py --phase select_pareto --help 
```

### Comparing the Frontiers

Finally, we can compare between the ground-truth and proxy Pareto frontiers, as follows:

```bash
python archai/nlp/search.py --phase gt_pareto --help
```

## Architecture Compression

Apart from finding more efficient architectures, it is also possible to compress current architectures and improve their efficiency without sacrificing their efficacy. Archai supports two types of architecture compression methods: ONNX and quantization.

### ONNX

The Open Neural Network Exchange (ONNX) format uses a graph with pre-defined operations implemented upon common standards. Additionally, several frameworks are implemented with such a format in mind, accelerating its inference and training, such as TensorRT and ONNXRuntime (ORT).

#### Exporting with ONNX

After the training has been conducted, it is relatively straightforward to produce an ONNX-based model, which defaults to the type of model shipped for production-ready environments. Archai provides an ONNX pipeline with custom classes and methods to allow the supported architectures to be better exported and optimized.

```bash
python archai/nlp/compression/onnx/export_torch_to_onnx.py --help
```

*When exporting a new model with ONNX, one can use the `--optimization` and `-quantization` arguments to enable graph optimization and quantization, reducing the model's footprint (less disk space due to int8 precision with less memory consumption).*

#### Validating with ONNX

With an ONNX-exported model, it is also essential to validate the export by comparing the outputs with its PyTorch counterpart version. Luckily, Archai offers ready-to-go scripts which allow users to check whether their ONNX model has been successfully exported.

```bash
python archai/nlp/compression/onnx/validate_onnx_export.py --help
```

```bash
python archai/nlp/compression/onnx/validate_past_key_values.py --help
```
 
Another validation application that might be worthwhile is [Netron](https://netron.app), which is essentially a graph visualizer.

### Quantization

Quantization stands for changing the precision of sets of operators from `float32` to `int8`. Commonly, such precision loss impacts performance and might hinder the desired outputs in particular tasks. In order to overcome such a problem, Archai also implements a Quantization Aware Training (QAT) pipeline, which concurrently calculates the optimal quantization parameters through a quantization simulation during training.

#### Post-Training Quantization (PTQ)

Post-Training Quantization is a straightforward post-training method, which quantizes operators from a pre-trained model in a dynamic fashion. In other words, scale factors and zero-points (quantization parameters) are dynamically based on the data range observed at runtime.

Note that such an approach is implemented by the `compression/quantization/ptq` module and can be applied in both [PyTorch](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/compression/quantization/ptq.py#L135) and [ONNX](https://github.com/microsoft/archai/blob/gderosa/lazy_loader/archai/nlp/compression/quantization/ptq.py#L109) models.

#### Quantization Aware Training (QAT)

On the other hand, if a model suffers significant performance degradation when being dynamically quantized, one can opt to use the Quantization Aware Training pipeline, which models the quantization errors in both forward and backward passes through simulated (fake) quantization modules. Essentially, the idea is to better estimate the scale factors and zero-points during training, which leads to a better-quantized model at the end of the procedure.

QAT with Archai is straightforward to be used, as it only requires a `--qat` or `--post_qat` flag to be employed in the training script. Nonetheless, if additional operators, observers, or quantizers need to be implemented, every QAT-related file can be found under the `compression/quantization` package.

## Metrics Scoring

Measuring the efficacy of auto-regressive (text-generation) architectures is not straightforward and often dependent on their losses/perplexity scores.

Thus, Archai, in partnership with the NLX team, has developed a scoring system (mostly NLX), denoted as `scoring_metrics`, to measure the efficacy of predictions. Overall speaking, the idea is to produce a set of predictions based on an input context and filter the most probable predictions according to a threshold, which is obtained through a linear interpolation over the validation dataset.

The `scoring_metrics/score` module provides a straightforward console application that can be invoked and tested out-of-the-box.

## Support

### Contributions

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Team

NLP with Archai is maintained by the Neural Architecture Search team in the [Reinforcement Learning](https://www.microsoft.com/en-us/research/group/reinforcement-learning-redmond) group of Microsoft Research at Redmond. We are available at all times to assist and provide more straightforward frameworks to conduct real-world experimentation.

### License

This project is released under the MIT License. Please review the [file](https://github.com/microsoft/archai/blob/master/LICENSE) for more details.

### Trademark

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
