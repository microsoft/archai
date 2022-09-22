# LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models

*This repository holds all the necessary code to run the very-same experiments described in the paper "LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```BibTex
@misc{Mojan:22,
  doi = {10.48550/ARXIV.2203.02094},
  url = {https://arxiv.org/abs/2203.02094},
  author = {Javaheripi, Mojan and Shah, Shital and Mukherjee, Subhabrata and Religa, Tomasz L. and Mendes, Caio C. T. and de Rosa, Gustavo H. and Bubeck, Sebastien and Koushanfar, Farinaz and Dey, Debadeepta},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

---

## Package Guidelines

### Installation

Install all the requirements using:

```Python
pip install -e .
```

*If you encounter any problems with the automatic installation of the [archai](https://github.com/microsoft/archai) package, contact us.*

### Data Loading and Utilities

#### Corpus

Corpus stands for a collection of pre-processed texts, which will be converted/trained into vocabularies and tokenizers. Although it is straightforward to add a new corpus, LTS uses the following ones provided by the `datasets/corpus` module:

* [WikiText-2 and WikiText-103](https://arxiv.org/abs/1609.07843);
* [1 Billion Word (lm1b)](http://www.statmt.org/lm-benchmark).

#### Vocabularies and Tokenizers

After defining the loaded data (dataset type), one can produce a new vocabulary and train a new tokenizer. There are several vocabulary/tokenization methods, depending on the task that will be employed or the model that will be used. Thus, LTS uses the following vocabularies and tokenizers from the `datasets/tokenizer_utils` package:

* [Word-based](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/tokenizer_utils/word_vocab.py);
* [GPT-2](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/tokenizer_utils/gpt2_vocab.py).

#### Iterators

The Causal Language Modeling (CLM) task aims to predict a `t+1` token based on the previous `t` tokens, i.e., predict a token followed by a sequence of tokens. In such a task, the model only attends to the left part of the context (previously seen tokens), often used by auto-regressive and text-generation models. LTS uses two types of CLM iterators, as follows:

* [Ordered iterator](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/lm_iterators.py#L7);
* [Multi-file iterator](https://github.com/microsoft/archai/blob/nlp/archai/nlp/datasets/lm_iterators.py#L177).


---

## Usage

### LiteTransformer Search

One can use the provided shell script to conduct every step needed to accomplish the search experimentation used throughout this paper, as follows:

```Bash
./archai/nlp/run_search.sh
```

*LTS comprises four steps: profiling the baseline scales, actual search, Pareto frontier selection, and baseline x Pareto frontier comparison.*

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, please do so! We will be available at our bests in this repository.

---