# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new tokens with a pre-trained model.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")
    parser.add_argument("hub_tokenizer_path", type=str, help="Path to the Hugging Face's Hub tokenizer.")
    parser.add_argument("prompt", type=str, help="Prompt to serve as the generation's context.")
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to a file where the generated text will be saved. If not specified, it will be printed to the console.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    full_path = max(Path(args.pre_trained_model_path).glob("checkpoint-*"), key=lambda x: int(x.stem.split("-")[-1]))
    model = AutoModelForCausalLM.from_pretrained(full_path).to(device)
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(args.hub_tokenizer_path)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=model.config.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=128,
    )

    if args.output_path:
        with open(args.output_path, "x") as f:
            f.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        print(f"Generated: \n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
