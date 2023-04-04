## Archai/Olive2 Integration

**Branch: lamorimfranc/olive-api**

Setting up the enviornment:

1. Create archai/devops environment with Python 3.8
    ```
	conda create -n <name> python=3.8
	conda activate <name>
    cd archai/snpe
    pip install -r requirements.txt
    ```

1. Install Olive2 from the local repository
    ```
    cd ..
    git clone https://aiinfra@dev.azure.com/aiinfra/PyTorch/_git/olive2
    cd olive2
    git checkout clovett/get_dlc_metrics
    pip install -r requirements.txt
    pip install -e .
    ```

1. Let Olive configure SNPE
    ```
	python -m olive.snpe.configure
    ```

    **If you run into a protobuf inconsistency with Python 3.8 you can workaround
    it by setting the folloiwng env. variable:**
    ```
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    ```
