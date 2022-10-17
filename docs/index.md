# Archai Documentation

**Archai** is a platform for Neural Network Search (NAS) that allows you to generate efficient deep networks for your applications. It offers the following advantages:

* 🔬 Easy mix-and-match between different algorithms;
* 📈 Self-documented hyper-parameters and fair comparison;
* ⚡ Extensible and modular to allow rapid experimentation;
* 📂 Powerful configuration system and easy-to-use tools.

## Initial Steps

To install the latest release:

```terminal
pip install archai
```

Please refer to the [installation guide](getting-started/install.md) for additional help.

```{note}
Archai requires Python 3.6+ and [PyTorch](https://pytorch.org) 1.2+.
```

### Toy Mode Experiment

The fastest way to try out the basic functionalities of Archai is to run a *toy mode* experiment using every available algorithm:

```terminal
python scripts/main.py
```

By switching to *toy mode*, algorithms will use tiny batches and a single epoch.

## Support and Contributions

Archai is maintained on [Github](https://github.com/microsoft/archai) by the Neural Architecture Search team of Microsoft Research at Redmond. Please feel free to:

* Open an [issue](https://github.com/microsoft/archai/issues) to request for support;
* Open a [pull request](https://github.com/microsoft/archai/pulls) to contribute with changes;
* Join the [Facebook](https://www.facebook.com/groups/1133660130366735) group to stay up-to-date.

## Citing Archai

If you use Archai in a scientific publication, please consider citing it:

```bibtex
@inproceedings{Archai:19,
    author = {Hu, Hanzhang and Langford, John and Caruana, Rich and Mukherjee, Saurajit and Horvitz, Eric J and Dey, Debadeepta},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    publisher = {Curran Associates, Inc.},
    title = {Efficient Forward Architecture Search},
    url = {https://proceedings.neurips.cc/paper/2019/file/6c468ec5a41d65815de23ec1d08d7951-Paper.pdf},
    volume = {32},
    year = {2019}
}
```

```{toctree}
---
hidden: true
caption: Getting Started
---

getting-started/install.md
getting-started/quickstart.md
getting-started/features.md
```

```{toctree}
---
hidden: true
caption: User Guide
---

user-guide/algorithms.md
user-guide/tutorial.md
user-guide/petridish.md
user-guide/food101.md
user-guide/deployment.md
```

```{toctree}
---
hidden: true
caption: Reference
---

reference/api.rst
reference/structure.md
reference/roadmap.md
reference/changelog.md
```

```{toctree}
---
hidden: true
caption: Support
---

support/faq.md
support/contact.md
support/copyright.md
```