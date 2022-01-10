# Natural Language Processing with Archai

Natural Language Processing (NLP) models use hardware advancements to solve more complex tasks. Nevertheless, such advancements also lead to an increased number of parameters, raising concerns regarding production-ready environments and low-resource devices.

Archai provides a straightforward alternative to find more efficient models through Neural Architecture Search (NAS), furnishing an ideal place to prototype and implement autoregressive transformer-based architectures. Essentially, the idea is to keep everything simple while offering developers and researchers every single tool to fulfill their needs.

Use NLP with Archai if you need a package or wish to:

* Fast-experiment transformer-based architectures;
* Design or use pre-loaded language modeling tasks;
* Increase your efficiency without losing effectiveness;
* Find new architectures under certain constraints.

## Table of contents

 * [Data Loading and Utilities](#data-loading-and-utilities)
    * [Corpus](#corpus)
    * [Vocabularies and Tokenizers](#vocabularies-and-tokenizers)
    * [Iterators](#iterators)
 * [Transformer-based Architectures](#transformer-based-architectures)
    * [Available Architectures](#available-architectures)
    * [Adding New Architectures](#adding-new-architectures)
    * [Training a New Model](#training-a-new-model)
 * [Neural Architecture Search](#neural-architecture-search)
    * [Evolutionary Search](#evolutionary-search)
    * [Finding the Pareto Frontier](#finding-the-pareto-frontier)
 * [Architecture Compression](#architecture-compression)
    * [ONNX Exporting](#onnx-exporting)
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

### Training a New Model

## Neural Architecture Search

### Evolutionary Search

### Finding the Pareto Frontier

## Architecture Compression

### ONNX Exporting

After the training has been conducted, it is relatively straightforward to produce an ONNX-based model, which defaults to the type of model shipped for production-ready environments. Archai provides an ONNX pipeline with custom classes and methods to allow the supported architectures to be better exported and optimized.

```bash
python archai/nlp/compression/onnx/export_torch_to_onnx.py --help
```

*When exporting a new model with ONNX, one can use the `--optimization` and `-quantization` arguments to enable graph optimization and quantization, reducing the model's footprint (less disk space due to int8 precision with less memory consumption).*

### Quantization

## Metrics Scoring

## Support

### Contributions

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Team

NLP with Archai is maintained by the [Reinforcement Learning](https://www.microsoft.com/en-us/research/group/reinforcement-learning-redmond) group of Microsoft Research at Redmond. We are available at all times to assist and provide more straightforward frameworks to conduct real-world experimentation.

### License

This project is released under the MIT License. Please review the [file](https://github.com/microsoft/archai/blob/main/LICENSE) for more details.

### Trademark

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.