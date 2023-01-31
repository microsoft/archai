<h1 align="center">
   <img src="https://user-images.githubusercontent.com/9354770/171523113-70c7214b-8298-4d7e-abd9-81f5788f6e19.png" alt="Archai logo" width="384px" />
   <br />
</h1>

<div align="center">
   <b>Archai</b> accelerates your Neural Architecture Search (NAS) through <b>fast</b>, <b>reproducible</b> and <b>modular</b> research, enabling the generation of efficient deep networks for various applications.
</div>

<br />

<div align="center">
	<img src ="https://img.shields.io/github/release/microsoft/archai?style=flat-square" alt="Release version" />
	<img src ="https://img.shields.io/github/issues-raw/microsoft/archai?style=flat-square" alt="Open issues" />
	<img src ="https://img.shields.io/github/contributors/microsoft/archai?style=flat-square" alt="Contributors" />
	<img src ="https://img.shields.io/pypi/dm/archai?style=flat-square" alt="PyPI downloads" />
	<img src ="https://img.shields.io/github/license/microsoft/archai?color=red&style=flat-square" alt="License" />
</div>

<br />

<div align="center">
   <a href="#installation">Installation</a> •
   <a href="#quickstart">Quickstart</a> •
   <a href="#tasks">Tasks</a> •
   <a href="#documentation">Documentation</a> •
   <a href="#support">Support</a>
</div>

## Installation

Archai can be installed through various methods, however, it is recommended to utilize a virtual environment such as `conda` or `pyenv` for optimal results.

To install Archai via PyPI, the following command can be executed:

```bash
pip install archai
```

**Archai requires Python 3.7+ and PyTorch 1.7.0+ to function properly.**

For further information, please consult the [installation guide](https://microsoft.github.io/archai/getting_started/installation.html).


## Quickstart

In this quickstart example, we will apply Archai in Natural Language Processing to find the optimal Pareto-frontier Transformers' configurations according to a set of objectives.

### Creating the Search Space

We start by importing the `TransformerFlexSearchSpace` class which represents the search space for the Transformer architecture:

```python
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import TransformerFlexSearchSpace

space = TransformerFlexSearchSpace("gpt2")
```

### Defining Search Objectives

Next, we define the objectives we want to optimize. In this example, we use `NonEmbeddingParamsProxy`, `TransformerFlexOnnxLatency`, and `TransformerFlexOnnxMemory` to define the objectives:

```python
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.nlp.parameters import NonEmbeddingParamsProxy
from archai.discrete_search.evaluators.nlp.transformer_flex_latency import TransformerFlexOnnxLatency
from archai.discrete_search.evaluators.nlp.transformer_flex_memory import TransformerFlexOnnxMemory

search_objectives = SearchObjectives()
search_objectives.add_objective(
   "non_embedding_params",
   NonEmbeddingParamsProxy(),
   higher_is_better=True,
   compute_intensive=False,
   constraint=(1e6, 1e9),
)
search_objectives.add_objective(
   "onnx_latency",
   TransformerFlexOnnxLatency(space),
   higher_is_better=False,
   compute_intensive=False,
)
search_objectives.add_objective(
   "onnx_memory",
   TransformerFlexOnnxMemory(space),
   higher_is_better=False,
   compute_intensive=False,
)
```

### Initializing the Algorithm

We use the `EvolutionParetoSearch` algorithm to conduct the search:

```python
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch

algo = EvolutionParetoSearch(
   space,
   objectives,
   None,
   "tmp",
   num_iters=5,
   init_num_models=10,
   seed=1234,
)
```

### Performing the Search

Finally, we call the `search()` method to start the NAS process:

```python
algo.search()
```

The algorithm will iterate through different network architectures, evaluate their performance based on the defined objectives, and ultimately produce a frontier of Pareto-optimal results.

## Tasks

To demonstrate and showcase the capabilities/functionalities of Archai, a set of end-to-end tasks are provided:

* [Text Generation](https://github.com/microsoft/archai/blob/main/tasks/text_generation).

## Documentation

The [official documentation](https://microsoft.github.io/archai) also provides a series of [notebooks](https://microsoft.github.io/archai/basic_guide/notebooks.html).

## Support

If you have any questions or feedback about the Archai project or the open problems in Neural Architecture Search, please feel free to contact us using the following information:

* Email: archai@microsoft.com
* Website: https://github.com/microsoft/archai/issues

We welcome any questions, feedback, or suggestions you may have and look forward to hearing from you.

### Team

Archai has been created and maintained by [Shital Shah](https://shital.com), [Debadeepta Dey](www.debadeepta.com), [Gustavo de Rosa](https://www.microsoft.com/en-us/research/people/gderosa), Caio Mendes, [Piero Kauffmann](https://www.microsoft.com/en-us/research/people/pkauffmann), Allie Del Giorno, Mojan Javaheripi, and [Ofer Dekel](https://www.microsoft.com/en-us/research/people/oferd) at Microsoft Research.

### Contributions

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademark

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

### License

This project is released under the MIT License. Please review the [file](https://github.com/microsoft/archai/blob/main/LICENSE) for more details.
