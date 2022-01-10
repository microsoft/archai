# Natural Language Processing with Archai

Natural Language Processing (NLP) models take advantage of hardware advancements to solve more complex tasks. Nevertheless, such advancements also lead to an increased number of parameters, raising concerns regarding production-ready environments and low-resource devices.

Archai provides a straightforward alternative to find more efficient models through Neural Architecture Search (NAS), furnishing an ideal place to prototype and implement autoregressive transformer-based architectures. Essentially, the idea is to keep everything simple while offering developers and researchers every single tool to fulfill their needs.

Use NLP with Archai if you need a package or wish to:

* Fast-experiment transformer-based architectures;
* Design or use pre-loaded language modeling tasks;
* Increase your efficiency without losing effectiveness;
* Find new architectures under certain constraints.

## Table of contents

 * [Data Loading and Utilities](#data-loading-and-utilities)
    * [Corpus](#corpus)
    * [Vocabularies](#vocabularies)
    * [Tokenizers](#tokenizers)
    * [Language Modeling Iterators](#language-modeling-iterators)
 * [Transformer-based Architectures](#transformer-based-architectures)
    * [NVIDIA's Memory Transformer](#nvidia's-memory-transformer)
    * [Huggingface's GPT-2](#huggingface's-gpt-2)
    * [Huggingface's Transformer-XL](#huggingface's-transformer-xl)
    * [Adding New Models](#adding-new-models)
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