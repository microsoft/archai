# Harness

## Installation

To install Harness, run the following commands in your command line:

```shell
conda create -n harness python=3.8
conda activate harness

pip install -e .
```

## Evaluating with LM-Evaluation-Harness (lm-eval)

To evaluate your model with LM-Evaluation-Harness, run the following command:

```shell
python evaluate_with_lm_eval.py --help
```

This will give you a list of options and arguments that can be passed to the script to evaluate your model. For example:

```shell
python evaluate_with_lm_eval.py gpt2 gpt2 --tasks cb,copa
```

This will evaluate a pre-trained GPT-2 from Hugging Face's Hub, using the `gpt2` pre-trained tokenizer on two SuperGLUE tasks: CommitmentBank and Choice of Plausible Alternatives.
