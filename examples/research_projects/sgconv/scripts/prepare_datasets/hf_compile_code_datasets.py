"""
Compiles a number of publicly available python code datasets
into a single dataset for the purpose of training code generation models.

The datasets this script requires:

1. codeparrot-train-near-deduplication -> 3556778 rows.
2. github-code-clean-python: just the python portion of github-code-clean -> 7177394 rows.
3. code-search-net-python: just the python portion of code-search-net's train, validation, test splits -> 457461 rows.
4. github-jupyter-text-code-pairs -> 770492 rows.
5. AhmedSoliman/CoNaLa-XLarge -> 32000 rows.
"""


import argparse
import os
from typing import List

from datasets import concatenate_datasets, load_dataset, load_from_disk


def process_codeparrot(examples):
    return {"code": examples["content"]}


def process_jupyter(example):
    concat_str = example["markdown"] + " \n " + example["code"]
    return {"code": concat_str}


def process_conalax(example):
    if example["snippet"] and example["intent"]:
        concat_str = example["intent"] + " \n " + example["snippet"]
        return {"code": concat_str}
    else:
        return {"code": ""}


def main():

    # load all the datasets
    codeparrot_dataset = load_from_disk("/data/codedatasets/codeparrot-train-near-deduplication")
    github_dataset = load_from_disk("/data/codedatasets/github-code-clean-python")
    codesearchnet_dataset = load_from_disk("/data/codedatasets/code-search-net-python")
    jupyter_dataset = load_from_disk("/data/codedatasets/github-jupyter-text-code-pairs")
    conalax_dataset = load_dataset("AhmedSSoliman/CoNaLa-XLarge")

    # process each dataset to extract
    # a common column name named "code"
    # which contains the code and text
    # strings the model will be trained on.

    # codeparrot
    codeparrot_clean = codeparrot_dataset["train"].map(
        process_codeparrot,
        batched=True,
        batch_size=1000,
        num_proc=1,
        with_indices=False,
        remove_columns=codeparrot_dataset["train"].column_names,
    )

    # github code
    remove_cols = set(github_dataset["train"].column_names)
    remove_cols.remove("code")
    github_clean = github_dataset["train"].remove_columns(remove_cols)

    # code search net
    codesearchnet_clean = codesearchnet_dataset.rename_column("whole_func_string", "code")
    remove_cols = set(codesearchnet_clean.column_names)
    remove_cols.remove("code")
    codesearchnet_clean = codesearchnet_clean.remove_columns(remove_cols)

    # jupyter
    jupyter_clean = jupyter_dataset["train"].map(
        process_jupyter,
        batched=False,
        batch_size=1,
        num_proc=1,
        with_indices=False,
        remove_columns=jupyter_dataset["train"].column_names,
    )

    conalax_clean_splits = []
    for split in {"train", "test", "validation"}:
        conalax_filtered = conalax_dataset[split].filter(
            lambda example: example["intent"] is not None and example["snippet"] is not None
        )
        conalax_clean = conalax_filtered.map(
            process_conalax,
            batched=False,
            batch_size=1,
            num_proc=1,
            with_indices=False,
            remove_columns=conalax_dataset[split].column_names,
        )
        conalax_clean_splits.append(conalax_clean)

    conalax_clean_splits.extend([codeparrot_clean, github_clean, codesearchnet_clean, jupyter_clean])

    public_python_dataset = concatenate_datasets(conalax_clean_splits)
    public_python_dataset.save_to_disk("/data/codedatasets/only_python")

    print("done")


if __name__ == "__main__":
    main()
