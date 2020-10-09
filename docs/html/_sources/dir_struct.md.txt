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
├── confs
├── dockers
├── docs
├── scripts
├── setup.py
├── tests
└── tools
    ├── azure
```

Archai core library resides in the [`archai`](https://github.com/microsoft/archai/tree/master/archai) folder.

[`nas`](https://github.com/microsoft/archai/tree/master/archai/nas) contains algorithm-agnostic infrastructure that is shared among NAS algorithms.

[`common`](archai/common) contains common non-NAS infrastructure code that can be used in other projects as well.

Algorithm-specific code resides in appropriately named folder like [`darts`](https://github.com/microsoft/archai/tree/master/archai/algos/darts), [`petridish`](https://github.com/microsoft/archai/tree/master/archai/algos/petridish), [`random`](https://github.com/microsoft/archai/tree/master/archai/algos/random), [`xnas`](https://github.com/microsoft/archai/tree/master/archai/algos/xnas).

[`scripts`](https://github.com/microsoft/archai/tree/master/scripts) contains entry-point script `main.py` as well as several other useful scripts.