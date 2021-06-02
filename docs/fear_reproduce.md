# Reproducing Experimental Results in [FEAR: Ranking Architectures by Feature Extraction Capabilities](anonymous)

Since `shortreg` and `FEAR` with different hyperparameters is not 
contained in Natsbench (Natsbench 201), the experiments require actually partially training 
1000 architectures sampled from Natsbench Topology search space. 
Consequently this requires significant compute. Here 
we detail the procedure to run the experiments but intentionally don't provide
wrapper scripts that will run all experiments and plot the results because
that is infeasible to finish on a single machine in a reasonable amount of time.
To support ease of reproduction, we will make public the associated 
log files upon publication.

## Install [Archai](https://github.com/microsoft/archai/tree/master/archai)
We utilize the open-source MIT licensed Archai NAS framework for the 
experiments in this work. Please follow the 
[installation instructions](https://github.com/microsoft/archai/blob/master/docs/install.md)
provided by the authors of the framework to install the latest version. 

## Download datasets
* Make a directory named `~/dataroot` and extract the below datasets in it.
  * [cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
  * [cifar100](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
  * [ImageNet16-120](https://image-net.org/download-images) 

## Reproducing figures 3, 6, 7, 8

In the paper figures 3, 6, 7, 8 represent the plots of average duration 
per architecture vs. Spearman’s correlation and 
average duration per architecture vs. common ratio over the top x% of the 
1000 architectures sampled from Natsbench topological search space on 
CIFAR10, CIFAR100, ImageNet16-120. We also show the various zero-cost 
measures from Abdelfattah et al. (22) in green. 

There are three sets of experiments to be run and their corresponding
logs processed and passed to a script that plots the figures.

* `shortreg`: 

The command line that will run `shortreg` (regular training of a neural network with shortened epochs):

```
python scripts/main.py \
--full \
--algos natsbench_regular_eval \
--common.seed 36 \
--nas.eval.loader.train_batch <batch_size> \
--nas.eval.trainer.epochs <num_epochs> \
--nas.eval.natsbench.arch_index <arch_id> \
--exp-prefix <exp_name> \
--datasets <datasets>
```

`--nas.eval.loader.train_batch <batch_size>` is the batch size to vary. For example on
CIFAR100 we vary batch size in 256,512,1024,2048.

`--nas.eval.trainer.epochs <num_epochs>` is the number of epochs of training to vary. 
For example on CIFAR100 we vary number of training epochs in 10,20,30.

`<arch_id>` is an architecture id in the list of 1000 uniform random architectures
sampled from Natsbench. For the exact list of architectures see the list in `main_proxynas_nb_wrapper.py`.
which also shows a simple way to distribute these 1000 architectures across machines.

`<exp_name>` is an appropriately chosen experiment name.

`<datasets>` is one of CIFAR10, CIFAR100, ImageNet16-120.

Each of the combinations above produces a folder with name `<exp_name>` containing 
corresponding log files. Each log must be analyzed by the analysis script:

```
python scripts/reports/fear_analysis/analysis_regular_natsbench_space.py \
--results-dir /path/to/exp_name \
--out-dir /path/to/processed/results
```

where `/path/to/processed/results` will be a folder created by the
script to save processed relevant data needed later on the for creating
plots over the 1000 architectures.

* `FEAR`

The command line that will run `FEAR` to evaluate each architecture:

```
python scripts/main.py \
--full \
--algos proxynas_natsbench_space \
--common.seed 36 \
--nas.eval.loader.freeze_loader.train_batch <freeze_batch_size> \
--nas.eval.trainer.epochs <num_epochs> \
--nas.eval.natsbench.arch_index <arch_id> \
--nas.eval.trainer.top1_acc_threshold <top1_acc_threshold> \
--exp-prefix <exp_name> \
--datasets <datasets>
```

`<freeze_batch_size>` is the batch size used for the second stage 
where most of the architecture is frozen and only the last few
layers are trained for a few more epochs.

`<top1_acc_threshold>` is the training accuracy threshold
up to which the entire network is trained before entering the 
second phase. This is dataset dependent and found by a shallow
pipeline. CIFAR10:0.6, CIFAR100:0.3, ImageNet16-120:0.2. 

Each of the combinations above produces a folder with name `<exp_name>` containing 
corresponding log files. Each log must be analyzed by the analysis script:

```
python scripts/reports/fear_analysis/analysis_freeze_natsbench_space_new.py
--results-dir /path/to/exp_name \
--out-dir /path/to/processed/results
```

where `/path/to/processed/results` will be a folder created by the
script to save processed relevant data needed later on the for creating
plots over the 1000 architectures.

* `zero cost` measures
  
The command line that will compute `zero cost` scores for each architecture:

```
python scripts/main.py \
--full \
--algos zerocost_natsbench_space \
--nas.eval.natsbench.arch_index <arch_id> \
--datasets <dataset>
```

Each of the combinations above produces a folder with name `<exp_name>` containing
corresponding log files. Each log must be analyzed by the analysis script:

```
python scripts/reports/fear_analysis/analysis_natsbench_zerocost.py \
--results-dir /path/to/exp_name \
--out-dir /path/to/processed/results
```

where `/path/to/processed/results` will be a folder created by the
script to save processed relevant data needed later on the for creating
plots over the 1000 architectures.









 

