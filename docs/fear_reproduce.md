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
Make a directory named `~/dataroot`


## Reproducing figures 3, 6, 7, 8

In the paper figures 3, 6, 7, 8 represent the plots of average duration 
per architecture vs. Spearman’s correlation and 
average duration per architecture vs. common ratio over the top x% of the 
1000 architectures sampled from Natsbench topological search space on 
CIFAR10, CIFAR100, ImageNet16-120. We also show the various zero-cost 
measures from Abdelfattah et al.(22) in green. 

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
--nas.eval.freeze_trainer.epochs <freeze_num_epochs> \
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

`freeze_num_epochs` is the number of epochs to train the network
in the second phase when most of the network is frozen.

Each of the combinations above produces a folder with name `<exp_name>` containing 
corresponding log files. Each log must be analyzed by the analysis script:

```
python scripts/reports/fear_analysis/analysis_freeze_natsbench_space_new.py
--results-dir /path/to/exp_name \
--out-dir /path/to/processed/results
```

where `/path/to/processed/results` will be a folder created by the
script to save processed relevant data needed later on for creating
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


* Collating all methods into a single plot:

Now that `shortreg`, `FEAR` and `zero-cost` measures have all been run and 
processed on the same set of 1000 architectures, one can use:

```
python /scripts/reports/fear_plots/cross_exp_plots.py \
--dataset <dataset_name>
--conf-location scripts/reports/fear_plots/cross_exp_conf.yaml
```

where `<dataset_name>` can take on values `natsbench_cifar10`, 
`natsbench_cifar100` or `natsbench_imagenet16-120` respectively.

`cross_exp_conf.yaml` has to be edited to input the exact names 
of various experiments used but should be pretty self-explanatory.

Note Table 2 in the paper is produced by manually inspecting
figures 3, 6, 7, 8 produced by the procedure above.

## Reproducing Table 3

Table 3 is produced by running `FEAR` and the `zero-cost` measures
on a dataset we term as Synthetic CIFAR10. So the first step is to 
reproduce this dataset. Note that since this dataset is produced
by a random process, we will make the exact instance used in the 
paper available upon acceptance and for the community to run further 
experiments on. Since this dataset is not part of any benchmark
we first fully train the 1000 architectures from Natsbench on this dataset
to produce *groundtruth* test accuracies. We keep the same 
training hyperparameters as used in Natsbench and train each architecture for  
200 epochs.

* Reproducing Synthetic CIFAR10.

Edit `out_dir` in `scripts/datasets/synthetic_gen/gen_synthetic_dataset.py` 
to reflect a path on disk you want to save the dataset in.

Then simply run `python scripts/datasets/synthetic_gen/gen_synthetic_dataset.py`
to generate the dataset.

* Fully training 1000 architectures on Synthetic CIFAR10.

```
python scripts/main.py \
--full \
--algos natsbench_regular_eval \
--common.seed 36 \
--nas.eval.loader.train_batch 256 \
--nas.eval.trainer.epochs 200 \
--nas.eval.natsbench.arch_index <arch_id> \
--exp-prefix <exp_name> \
--datasets synthetic_cifar10
```

followed by an analysis script on the log files generated
by the full training:

```
python scripts/reports/fear_analysis/analysis_natsbench_nonstandard_generate_benchmark.py \
--results-dir /path/to/logs/from/full/training
--out-dir /path/to/folder/for/saving/benchmark
```

This will generate a file named `archid_test_accuracy_synthetic_cifar10.yaml` 
which contains for every architecture id in the set of 1000 used, the test 
accuracy it obtained on this synthetic dataset. This file is then passed in 
to downstream analysis scripts as detailed below.

* `zero-cost` on Synthetic CIFAR10.

Same as running zero-cost measures on any other dataset:

```
python scripts/main.py \
--full \
--algos zerocost_natsbench_space \
--datasets synthetic_cifar10
```

* `FEAR` on Synthetic CIFAR10.

Same as running `FEAR` on any other dataset

```
python scripts/main.py \
--full \
--algos proxynas_natsbench_space \
--common.seed 36 \
--nas.eval.loader.freeze_loader.train_batch 1024 \
--nas.eval.freeze_trainer.epochs 5 \
--nas.eval.natsbench.arch_index <arch_id> \
--nas.eval.trainer.top1_acc_threshold 0.15 \
--exp-prefix <exp_name> \
--datasets synthetic_cifar10
```

Each of the combinations above produces a folder with name `<exp_name>` containing 
corresponding log files. Each log must be analyzed by the analysis script:

```
python scripts/reports/fear_analysis/analysis_freeze_natsbench_space_new.py
--results-dir /path/to/exp_name \
--out-dir /path/to/processed/results \
--reg-evals-file /path/to/archid_test_accuracy_synthetic_cifar10.yaml
```

where `/path/to/processed/results` will be a folder created by the
script to save processed relevant data needed later on for creating
plots over the 1000 architectures. Note the use of `archid_test_accuracy_synthetic_cifar10.yaml`
since this dataset is not part of the Natsbench benchmark.

* Collating all methods into a single plot:
Now that ranking methods and the full training have been run, the 
plots comparing all the methods can be generated using the same 
process and scripts as for benchmark datasets like 
CIFAR10, CIFAR100 detailed above.


## Reproducing Random Search Results

* Random Search with FEAR 

```
python scripts/main.py \
--full \
--algos random_natsbench_tss_far \
--datasets <dataset_name> \
--nas.search.trainer.top1_acc_threshold <dataset_specific_threshold> \
--nas.search.max_num_models 500 \
--nas.search.ratio_fastest_duration 4 \
--common.seed <seed> \
--no-eval
```

Analysis:
```
python /scripts/reports/fear_analysis/analysis_random_search_natsbench_tss_far.py \
--results-dir /path/to/results \
--out-dir /path/to/save/dir
```

* Random Search with `shortreg`

```
python scripts/main.py \
--full \
--algos random_natsbench_tss_reg \
--datasets <dataset_name> \
--nas.search.max_num_models 500 \ 
--common.seed <seed> \
--no-eval
```

Analysis:

```
python /scripts/reports/fear_analysis/analysis_random_search_natsbench_tss_reg.py \
--results-dir /path/to/results \
--out-dir /path/to/save/dir
```




























