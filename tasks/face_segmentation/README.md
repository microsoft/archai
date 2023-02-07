# Face Segmentation

## Face Synthetics
![Face Synthetics](assets/face_synthetics.png)

## Search Space

The search space used for this example is based on the [Stacked HourglassNet architecture](https://arxiv.org/abs/1603.06937). 

![HourglassNet search space](assets/search_space.png)



## Search

To run a search job, use the following command

```shell
python3 --dataset_dir FACE_SYNTHETICS_DIR --output_dir output_dir
```

To change the search parameters, either change the file in (confs/search_config.yaml)[confs/search_config.yaml] or create a new one and set `--search_config MY_NEW_CONFIG.yaml`.
By default, `search.py` will run multiple partial training jobs using Ray (2 jobs per GPU). To change the number of gpus per job, set `--gpus_per_job`, or use the `--serial_training` flag to disable parallel training jobs altogether.

### Search Results

![pareto_evolution](assets/pareto_evolution.png)

The selected architectures can be found inside the [archs/](archs/) folder or in the table below:


| Architecture                             	| One Epoch Validation IOU 	| ONNX Latency (ms) 	| Search iteration 	|
|------------------------------------------	|--------------------------	|-------------------	|------------------	|
| [d650d48bdc83e75ae6ace9f20c17caa65fb5048a](archs/d650d48bdc83e75ae6ace9f20c17caa65fb5048a.json) 	| 0.784                    	| 0.070              	| 9                	|
| [07d670b8f76d8b9ca1e39379798d8b0046695b6a](archs/07d670b8f76d8b9ca1e39379798d8b0046695b6a.json) 	| 0.769                    	| 0.035             	| 6                	|
| [0cf2105627cd8ef8b86bdafd4987714dc2519827](archs/0cf2105627cd8ef8b86bdafd4987714dc2519827.json) 	| 0.731                    	| 0.027             	| 8                	|
| [b049ce7b41268d956af5410a3e838a2992d29232](archs/b049ce7b41268d956af5410a3e838a2992d29232.json) 	| 0.711                    	| 0.026             	| 4                	|
| [f22f089ae8c618117f4869f20213b344189bab9a](archs/f22f089ae8c618117f4869f20213b344189bab9a.json) 	| 0.710                    	| 0.025             	| 4                	|
| [1531903d654ecc930a0659e31b42c3efe6fe6ef3](archs/1531903d654ecc930a0659e31b42c3efe6fe6ef3.json) 	| 0.705                    	| 0.022             	| 6                	|
| [31cc57fe423f06a0f4d6ba000fe1e3decd3a442c](archs/31cc57fe423f06a0f4d6ba000fe1e3decd3a442c.json) 	| 0.686                    	| 0.019             	| 8                	|
| [1ab34d5fb31ef986650b5b112cfa3eca104b8107](archs/1ab34d5fb31ef986650b5b112cfa3eca104b8107.json) 	| 0.683                    	| 0.018             	| 8                	|
| [0c74d6d48a3514be3e80a84593c5f6b3f656fb3c](archs/0c74d6d48a3514be3e80a84593c5f6b3f656fb3c.json) 	| 0.681                    	| 0.016             	| 8                	|
| [1f1a7d04c4925d17f0575418cc974327ab71a93a](archs/1f1a7d04c4925d17f0575418cc974327ab71a93a.json) 	| 0.666                    	| 0.015             	| 8                	|
| [e6b8640bd2b83212e3256907a2382ae9bb799b65](archs/e6b8640bd2b83212e3256907a2382ae9bb799b65.json) 	| 0.654                    	| 0.012             	| 5                	|
| [82419a2ad358a34c508444c86db261616cf45ec3](archs/82419a2ad358a34c508444c86db261616cf45ec3.json) 	| 0.620                    	| 0.011             	| 3                	|
| [15914e86631373b2d9c823873ba6a88a1dc548c7](archs/15914e86631373b2d9c823873ba6a88a1dc548c7.json) 	| 0.606                    	| 0.010              	| 9                	|
| [de9067fa95074057353c67f62036a5b395a2d6a2](archs/de9067fa95074057353c67f62036a5b395a2d6a2.json) 	| 0.554                    	| 0.009             	| 8                	|
| [be543f6a3d1eadc9a42496f0b40871d82d4931df](archs/be543f6a3d1eadc9a42496f0b40871d82d4931df.json) 	| 0.516                    	| 0.007             	| 8                	|


## Final Training

To run complete training for a chosen architecture, use the `train.py` script

```shell
python3 train.py [path_to_final_architecture] --dataset_dir FACE_SYNTHETICS_DIR --output_dir output_dir --epochs n_epochs
```

The pareto architecture files selected by the search algorithm can be found under `SEARCH_OUT_DIR/pareto_models_iter_XX`. A table with the partial training performance and other objectives can be found in the `SEARCH_OUT_DIR/search_state_XX.csv` file.
