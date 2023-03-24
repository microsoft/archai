# Evaluating Models on HumanEval

This guide will provide step-by-step instructions to install the required dependencies and evaluate pre-trained models on HumanEval.

## Installing Dependencies

To begin, please install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Evaluating a DeepSpeed Model

If you are evaluating a pre-trained DeepSpeed model, the process is different from evaluating with Hugging Face. Unlike Hugging Face, DeepSpeed does not have the benefit of `model.from_pretrained()`. Therefore, we need to load the DeepSpeed checkpoint, gather the model's state, and apply it to the model instance. To evaluate a pre-trained DeepSpeed model, run the following command:

```bash
python deepspeed/evaluate_human_eval.py --help
```

## Evaluating a Hugging Face Model

If you are evaluating a pre-trained Hugging Face model, the only requirement is that the checkpoint folder follows the Hugging Face format, which includes a folder named `checkpoint-step_number` and is composed of `config.json` and `*.pt` files. To evaluate a pre-trained Hugging Face model, run the following command:

```bash
python hf/evaluate_human_eval.py --help
```

*The `--help` argument will prompt the helper and provide a description of each argument.*
