# Evaluation

Please install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## DeepSpeed

Loading a pre-trained DeepSpeed model is different from HF and do not have the benefit of `CodeGenForCausalLM.from_pretrained()`. Thus, we need to load the DeepSpeed checkpoint, gather the model's state and apply to the model instance.

The full checkpoint restoration + evaluation script can be run using the following command:

```bash
python deepspeed/evaluate_human_eval.py
```

## Hugging Face

You can evaluate pre-trained checkpoints using HumanEval. The only requirement is that the checkpoint folder follows the Hugging Face format, i.e., has a folder named `checkpoint-step_number` and is composed of `config.json` and `*.pt` files:

```bash
python hf/evaluate_human_eval.py
```
