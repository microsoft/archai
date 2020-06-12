# Directory Structure

```
├── archai
│   ├── cifar10_models
│   ├── common
│   ├── darts
│   ├── data_aug
│   ├── nas
│   ├── networks
│   ├── petridish
│   ├── random
│   └── xnas
├── archived
├── confs
├── dockers
├── docs
├── scripts
├── setup.py
├── tests
└── tools
    ├── azure
```

Most of the functionality resides in the [`archai`](archai/) folder.
[`nas`](archai/nas) contains algorithm-agnostic infrastructure
that is commonly used in NAS algorithms. [`common`](archai/common) contains
common infrastructure code that has no nas-specific code but is infrastructre
code that gets widely used.
Algorithm-specific code resides in appropriately named folder like [`darts`](archai/nas/darts),
[`petridish`](archai/nas/petridish), [`random`](archai/nas/random),
[`xnas`](archai/nas/xnas)

[`scripts`](archai/scripts) contains entry-point scripts to running all algorithms.