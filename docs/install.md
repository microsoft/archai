# Installing Archai

## Prerequisites

Archai requires Python 3.6+ and [PyTorch](https://pytorch.org/get-started/locally/) 1.2+. To install Python we highly recommend [Anaconda](https://www.anaconda.com/products/individual#Downloads). Archai works both on Linux as well as Windows.

## Install from source code

We recommend installing from the source code:

```bash
git clone https://github.com/microsoft/archai.git
cd archai
install.sh # on Windows, use install.bat
```

## Using Dockers

You can also use dockers in the [dockers](https://github.com/microsoft/archai/tree/master/dockers) folder. These are useful for large scale experimentation on compute clusters.

## Notes

Vast majority of Archai code base works by simply by installing the package by running the command `pip install -e .` in archai source directory. However, this won't install pydot and graphviz to visualize the generated architectures. Also, if you have older pickle5 versions, you might get some errors which can be resolved by installing pickle5 using conda. This is what [install script](https://github.com/microsoft/archai/blob/master/install.sh) does.
