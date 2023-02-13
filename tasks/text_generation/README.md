# Text Generation

At Archai, we understand the significance of finding the optimal neural architecture in order to achieve the highest performance in text generation. That's why we have developed a cutting-edge neural architecture search method named Lightweight Transformer Search (LTS). This innovative method allows us to find the most optimal architectures that lie on the Pareto Frontier, where trade-offs are made between multiple objectives such as latency and memory usage.

## Model Gallery

We utilized GPT-2 as our base model and applied LTS on top of it to find the best performing architectures given a set of constraints. The following table showcases the results of our search:

| Model | Non-Embedding Parameters (M) | Latency (s) | Memory (MB) |
| - | - | - | - |
[gpt2_a9e3147996070fda25af4b39ed95b6a18d6d0402](https://github.com/microsoft/archai) | 1.06 | 0.008 | 29.06
[gpt2_80fabe4acddff0dc796e287588e40d86e79df4b2](https://github.com/microsoft/archai) | 2.08 | 0.013 | 45.46
[gpt2_90682823835acabd965294775983a1d5a2c2fa43](https://github.com/microsoft/archai) | 3.13 | 0.021 | 74.50
[gpt2_c76bdddb5cf59275711672daa5b8c70e6c78bf4e](https://github.com/microsoft/archai) | 3.95 | 0.024 | 77.62
[gpt2_8f5159304179c77ecdc69c953b71a3f8fa528564](https://github.com/microsoft/archai) | 5.13 | 0.030 | 94.64
[gpt2_131845381012a68c3a358514fdffc12b09db1ed8](https://github.com/microsoft/archai) | 6.44 | 0.036 | 112.16
[gpt2_917c2f9601a1c29d1f280bb172015e5fb210b6b3](https://github.com/microsoft/archai) | 7.41 | 0.042 | 90.76
[gpt2_538d4b101df48595a935d90dbf4a7fb2ac09ac01](https://github.com/microsoft/archai) | 8.23 | 0.047 | 93.88
[gpt2_c679fa01f00dd6f584614c6d9784eb233b047283](https://github.com/microsoft/archai) | 9.46 | 0.053 | 148.71
[gpt2_39563367097004cfd771d76d8822e51ad79b56d6](https://github.com/microsoft/archai) | 10.65 | 0.051 | 190.77
[gpt2_ddf63c1125f1fed5a7dd3537f640834187719996](https://github.com/microsoft/archai) | 13.32 | 0.069 | 125.78
[gpt2_0e1b5a3c867d6473da270799061f3089a1df5afd](https://github.com/microsoft/archai) | 16.04 | 0.084 | 173.74
[gpt2_3b30c85ac08c6b12b0ea46cb832270ba52b7fcd8](https://github.com/microsoft/archai) | 18.97 | 0.096 | 209.94
[gpt2_1e9d92f0fed7288facc68cb448863e8120ccca9c](https://github.com/microsoft/archai) | 20.96 | 0.105 | 217.50
[gpt2_0e8c86e6babd924ff8b511c94cc1647bf61f81a2](https://github.com/microsoft/archai) | 24.83 | 0.121 | 244.77
[gpt2_5fea22df661ad91676709da7a334505f15765659](https://github.com/microsoft/archai) | 26.89 | 0.131 | 252.65
[gpt2_46e7c68a025417e20a7e13bd4c1ee71438d28069](https://github.com/microsoft/archai) | 30.07 | 0.146 | 252.23
[gpt2_98b0196b5a865ba76f31723646f33e0461dc910d](https://github.com/microsoft/archai) | 33.24 | 0.160 | 314.39
[gpt2_4352a56f3fa9e7ba6d291867d356a08022753658](https://github.com/microsoft/archai) | 40.34 | 0.195 | 328.88
[gpt2_6c6e63116ff74ba444ff5a08cef54380073ebea3](https://github.com/microsoft/archai) | 49.85 | 0.230 | 377.68

## Searching for Pareto-optimal Architectures

We ran LTS for a total of 10 generations and discovered multiple architectures that perform well with regards to non-embedding parameters, latency, and memory. To reproduce the search, the following command can be used:

```python
python search.py -h
```

*The arguments used on this task are the default ones provided by the script.*

### Results

The points to the bottom-left of the plot indicate the best architectures in terms of non-embedding parameters and ONNX-based latency.

![Non-Embedding Parameters x ONNX Latency Plot](assets/pareto_non_embedding_params_vs_onnx_latency.png)

The points to the bottom-left of the plot represent the best architectures in terms of non-embedding parameters and ONNX-based memory.

![Non-Embedding Parameters x ONNX Memory Plot](assets/pareto_non_embedding_params_vs_onnx_memory.png)

## Training the Architectures

Once the Pareto-optimal architectures have been found (located in the `models` folder), they can be trained using the following script:

```python
python train.py -h
```

*The arguments used on this task are the default ones provided by the script. The dataset used for training is 7.8B tokens from a pre-encoded version of ThePile.*

### Results

## Generating Text with Pre-Trained Architectures

With our pre-trained architectures, high-quality text can be generated with ease using just a few lines of code. Simply download one of the models from our Model Gallery and start generating text immediately:

```python
python generate_text.py -h
```

As an alternative, one can use models from Hugging Face's Hub to generate text, such as:

```python
python generate_text.py "gpt2" "gpt2" <prompt>
```
