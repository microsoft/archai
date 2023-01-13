# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import torch
from transformers import AutoModelForCausalLM, StoppingCriteriaList

from archai.nlp.datasets.hf.tokenizer_utils import ArchaiPreTrainedTokenizerFast
from archai.nlp.eval.harness import MultipleTokenStoppingCriteria

# Stop-tokens used to stop the generation
STOP_TOKENS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new tokens with a pre-trained model.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    parser.add_argument(
        "hub_tokenizer_path",
        type=str,
        help="Name or path to the Hub's tokenizer.",
    )

    parser.add_argument("prompt", type=str, help="Prompt to serve as the generation's context.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained(args.hub_tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(args.pre_trained_model_path).to(device)
    model.config.use_cache = True

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    stop_tokens = tokenizer(STOP_TOKENS, return_tensors="pt", padding="longest").input_ids.to(device)

    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=model.config.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        stopping_criteria=StoppingCriteriaList([MultipleTokenStoppingCriteria(stop_tokens)]),
    )

    print(f"Generated: \n {tokenizer.decode(output[0], skip_special_tokens=True)}")
