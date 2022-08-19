# Setting up Azure ML

If you are not familiar with Azure ML workspaces, it will be useful to go through
these [tutorials](https://docs.microsoft.com/en-us/azure/machine-learning/) first. In particular
the usage of [Python SDK](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup)

* Push local dockers to AML workspace registries. See these [instructions](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli).

* Copy over `aml_config_template.yaml` to outside the repository, rename appropriately example `aml_config_myname.yaml` and fill-in secret keys like your Azure subscription keys, storage account details, registry details etc. NOTE: It is important that you not accidentally add this to version control. Once this is in version control even if you delete from the working copy all the secrets are still in history and it takes a lot of work to delete all evidence from history. If the repo is made open source with the secrets in history, it will pose a security issue.

* On command prompt `python tools/azure/aml_launch_main.py --aml_secrets_filepath /path/to/aml_secrets_myname.yaml --algo darts --full` to launch a full darts search and evaluation on Azure AML. Modify the command line appropriately for other algorithms.
