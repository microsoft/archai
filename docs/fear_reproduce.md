# Reproducing Experimental Results in [FEAR: Ranking Architectures by Feature Extraction Capabilities](anonymous)

Since `shortreg` and `FEAR` with different hyperparameters is not 
contained in Natsbench, the experiments require actually partially training 
1000 architectures sampled from Natsbench Topology search space. 
Consequently this requires significant compute. To support ease of reproduction,
we will also make public the associated log files upon publication.

## Install [Archai](https://github.com/microsoft/archai/tree/master/archai)
We utilize the open-source MIT licensed Archai NAS framework for the 
experiments in this work. Please follow the 
[installation instructions](https://github.com/microsoft/archai/blob/master/docs/install.md)
provided by the authors of the framework to install the latest version. 

## Download datasets
Make a directory named `~/dataroot`.
Download [cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 
and [cifar100](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) from 


## Reproducing figures 3, 6, 7, 8

In the paper figures 3, 6, 7, 8 represent the plots of average duration 
per architecture vs. Spearman’s correlation and 
average duration per architecture vs. common ratio over the top x% of the 
1000 architectures sampled from Natsbench topological search space on 
CIFAR10, CIFAR100, ImageNet16-120. We also show the various zero-cost 
measures from Abdelfattah et al.(22) in green. 

There are three sets of experiments to be run and their corresponding
logs processed and passed to a script that plots the figures.

1. `shortreg`: 


