# Archai Features

Archai is designed to unify several latest algorithms for Network Architecture Search into a common codebase allowing for much more generality as well as reproducibility with fair comparison. While enabling next generation research in NAS, we also aim to provide high quality turn-key implementations to rapidly try out these algorithms on your custom datasets and scenarios. This page describes several high level features available in Archai.

## Features

* **Declarative Approach and Reproducibility**: Archai carefully abstracts away various hyperparameters, training details, model descriptions etc into a configuration. The goal is to make several critical decisions explicit that otherwise may get burried in the code making it harder to reproduce experiment. In addition, Archai's configuration ssytem goes well beyond standard yaml by ability to inherit from config from one experiment and make only few changes or override configs from command line. It is possible to perform several different experiments  by merely changing config that otherwise might need significant code changes.

* **Plug-n-Play Datasets**: Archai provides infrastructure so a new dataset can be added by simply adding new config. This allows for much faster experimentation for real-world datasets and leverage latest NAS research to benefit actual products.

* **