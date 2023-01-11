import token
import tokenize
from io import BytesIO
from tokenize import TokenInfo
from typing import List

import numpy as np
import plotly.graph_objects as go
from datasets import load_dataset, load_from_disk
from src.python_tokenizer import PreTrainedTokenizerFastForPython
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    pipeline,
)


def find_postfixes(python_tokens: List[TokenInfo]) -> List[str]:
    """For a list of python tokens calculates
    the separator after each token from
    position information.
    """
    sep_strs = []
    for i in range(len(python_tokens)):
        pair = python_tokens[i : i + 2]
        if len(pair) == 1:
            # end of list, just insert a space
            sep_strs.append(" ")
        else:
            a, b = pair
            if a.start[0] == b.start[0]:
                # if both a and b on the same line
                # then process as usual
                if b.start[1] <= a.end[1]:
                    # no space
                    sep_strs.append("")
                else:
                    # insert a space
                    # TODO: might need to handle multiple spaces
                    sep_strs.append(" ")
            else:
                # if they are on different lines
                # then no space
                sep_strs.append("")

    assert len(sep_strs) == len(python_tokens)
    return sep_strs


def inject_python_types(examples: List[str]) -> str:
    injected_strs = []
    for codestr in examples:

        try:
            tokens = tokenize.tokenize(BytesIO(codestr.encode("utf-8")).readline)
            python_tokens = list(tokens)
        except Exception:
            injected_strs.append(None)
            continue

        python_types = [token.tok_name[t.type] for t in python_tokens]
        sep_strs = find_postfixes(python_tokens)
        string_accum = ""
        for tok, sep, ptype in zip(python_tokens[1:], sep_strs[1:], python_types[1:]):
            # sometimes there are parse fails
            # so ignoring those
            if ptype != "ERRORTOKEN":
                string_accum += f"[s_{ptype}]" + tok.string + f"[e_{ptype}]"
            else:
                string_accum += tok.string
            string_accum += sep
        injected_strs.append(string_accum)
    return {"code": injected_strs}


def main():

    datasetname = "/data/codedatasets/only_python_dedup"
    dataset = load_from_disk(datasetname)
    print(f"using dataset of length {dataset.num_rows}")

    # # tokenize example code string to check
    # codestr = dataset['code'][100000]
    # python_string = inject_python_types([codestr])

    injected_dataset = dataset.map(
        inject_python_types,
        input_columns="code",
        batched=True,
        batch_size=1000,
        num_proc=62,
        with_indices=False,
    )

    filtered_dataset = injected_dataset.filter(lambda example: example["code"] is not None)

    filtered_dataset.save_to_disk("/data/codedatasets/only_python_dedup_injected")

    print("done.")


if __name__ == "__main__":
    main()
