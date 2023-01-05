import numpy as np
from datasets import load_from_disk


def main():
    enc_dataset = load_from_disk("/data/codedatasets/only_python_dedup_encoded_2048")

    # check the maximum id produced by the tokenizer
    # and if it is exceeding 50257 since that is the vocab_size
    # that the "Salesforce/codegen-350M-mono" claims.

    ID_MAX = 50257

    exceeding_ids = []

    def check_token_ids(examples):
        input_ids = np.array(examples["input_ids"])
        exceeding_ids.append(np.max(input_ids[input_ids > ID_MAX]))

    _ = enc_dataset.map(check_token_ids, batched=True, batch_size=1000, with_indices=False, num_proc=1)

    print("done.")


if __name__ == "__main__":
    main()
