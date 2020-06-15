# Welcome to Archai

Archai is a platform for Neural Network Search (NAS) with a goal to unify several recent advancements in research
and making them accessible to non-experts so that anyone can leverage this research to generate efficient deep networks for their own applications. Archai hopes to accelerate NAS research by easily allowing to mix and match different techniques rapidly while still ensuring reproducibility, documented hyper-parameters and fair comparison across the spectrum of these techniques. Archai is extensible and modular to accommodate new algorithms easily (often with only a few new lines of code) offering clean and robust codebase.

[Extensive feature list](docs/features.md)

## How to Get It

### Install as package

```
pip install archai
```

### Install from source code

We recommend installing from the source code:

```
git clone https://github.com/microsoft/archai.git
cd archai
pip install -e .
```


Archai requires Python 3.6+ and is tested with PyTorch 1.3+. For network visualization, you may need to separately install [graphviz](https://graphviz.gitlab.io/download/). We recommand


## Test installation

* `cd archai`
* The below command will run every algorithm through a few batches of cifar10
  and for both search and final training
* `python scripts/main.py`. If all went well, you have a working installation! Yay! 
* Note one can also build and use the cuda 10.1 or 9.2 compatible dockers
  provided in the [dockers](dockers) folder. These dockers are useful
  for large scale experimentation on compute clusters.

## How to Use It

### Quick start

[`scripts/main.py`](archai/scripts/main.py) is the main point of entry.

#### Run all algorithms in toy mode

`python scripts/main.py` runs all implemented search algorithms and final training
with a few minibatches of data from cifar10. This is designed to exercise all
code paths and make sure that everything is properly.

#### To run specific algorithms

`python scripts/main.py --darts` will run darts search and evaluation (final model training) using only a few minibatches of data from cifar10.
`python scripts/main.py --darts --full` will run the full search.

Other algorithms can be run by specifying different algorithm names like `petridish`, `xnas`, `random` etc.

#### List of algorithms

Current the following algorithms are implemented:

* [Petridish](https://papers.nips.cc/paper/9202-efficient-forward-architecture-search.pdf)
* [DARTS](https://deepmind.com/research/publications/darts-differentiable-architecture-search)
* [Random search baseline]
* [XNAS](http://papers.nips.cc/paper/8472-xnas-neural-architecture-search-with-expert-advice.pdf) (this is currently experimental and has not been fully reproduced yet as XNAS authors have not released source code at the time of writing.)
* [DATA](https://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation.pdf) (this is currently experimental and has not been fully reproduced yet as XNAS authors have not released source code at the time of writing.)

See [Roadmap](#roadmap) for details on new algorithms coming soon.

### Tutorials

### Running experiments on Azure AML

See detailed [instructions](tools/azure/README.md).

### Other References

* [Directory Structure](docs/dir_struct.md)
* [FAQ](docs/faq.md)
* [Roadmap](docs/roadmap.md)

## Contribute

We would love your contributions, feedback, questions, algorithm implementations and feature requests! Please [file a Github issue](https://github.com/microsoft/archai/issues/new) or send us a pull request. Please review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [learn more](https://github.com/microsoft/archai/blob/master/CONTRIBUTING.md).

## Contact

Join the Archai group on [Facebook](https://www.facebook.com/groups/1133660130366735/) to stay up to date or ask any questions.

## Team
Archai has been created and maintained by [Shital Shah](https://shitalshah.com) and [Debadeepta Dey](www.debadeepta.com) in the [Reinforcement Learning Group](https://www.microsoft.com/en-us/research/group/reinforcement-learning-redmond/) at Microsoft Research AI, Redmond, USA. 

They look forward to Archai becoming more community driven and including major contributors here. 

## Credits

Archai builds on several open source codebases. These includes: [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment), [pt.darts](https://github.com/khanrc/pt.darts), [DARTS-PyTorch](https://github.com/dragen1860/DARTS-PyTorch), [DARTS](https://github.com/quark0/darts), [petridishnn](https://github.com/microsoft/petridishnn), [PyTorch CIFAR-10 Models](https://github.com/huyvnphan/PyTorch-CIFAR10), [NVidia DeepLearning Examples](https://github.com/NVIDIA/DeepLearningExamples), [PyTorch Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr). Please see `install_requires` section in [setup.py](setup.py) for up to date dependencies list. If you feel credit to any material is missing, please let us know by filing a [Github issue](https://github.com/microsoft/archai/issues/new).

## License

This project is released under the MIT License. Please review the [License file](LICENSE.txt) for more details.
