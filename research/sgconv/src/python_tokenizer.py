# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import token
import tokenize
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerFast

from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)


class PreTrainedTokenizerFastForPython(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, texts: List[str]) -> Dict[str, Any]:
        results = {
            "group_ids": [],
            "python_ids": [],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "raw_tokens": [],
        }

        # TODO: Apply `multiprocessing` for speed up
        for text in texts:
            try:
                # Uses Python tokenizer and ignores the encoding string
                tokens = tokenize.tokenize(BytesIO(text.encode("utf-8")).readline)
                python_tokens = list(tokens)[1:]
            except Exception:
                logger.warning(f"text: {text} could not be parsed.")

                results["group_ids"].append([])
                results["python_ids"].append([])
                results["input_ids"].append([])
                results["token_type_ids"].append([])
                results["attention_mask"].append([])
                results["raw_tokens"].append([])

                continue

            group_ids, python_ids, input_ids, raw_tokens = [], [], [], []
            name_counter = defaultdict(int)

            counter = 0
            for t in python_tokens:
                codegen_tokens = self.tokenize(t.string)
                codegen_ids = self.encode(t.string)

                # Annotates CodeGen tokens as same python token and type information
                raw_tokens.extend(codegen_tokens)
                input_ids.extend(codegen_ids)

                # Re-uses group_ids to look up for same symbols during network's training
                python_ids.extend([token.tok_name[t.type]] * len(codegen_tokens))
                if token.tok_name[t.type] == "NAME" and t.string in name_counter:
                    group_id = name_counter[t.string]
                elif token.tok_name[t.type] == "NAME":
                    name_counter[t.string] = counter
                    group_id = counter
                else:
                    group_id = counter

                group_ids.extend([group_id] * len(codegen_tokens))
                counter += 1

            # setting token_type_ids for compatibility
            # may not need this
            token_type_ids = [0] * len(group_ids)

            # setting attention mask to all 1's
            # since we are not handling padding here yet.
            attention_mask = [1] * len(group_ids)

            assert len(group_ids) == len(python_ids)
            assert len(group_ids) == len(input_ids)
            assert len(group_ids) == len(token_type_ids)
            assert len(group_ids) == len(attention_mask)
            assert len(group_ids) == len(raw_tokens)

            # replace with delimiters
            dups = set()
            textcopy = text
            for i in range(len(python_ids)):
                python_id = python_ids[i]
                string_token = self.decode(input_ids[i])
                # replace with delimiters in string
                if string_token not in dups:
                    textcopy = textcopy.replace(string_token, python_id + string_token + python_id)
                    dups.add(string_token)

            results["group_ids"].append(group_ids)
            results["python_ids"].append(python_ids)
            results["input_ids"].append(input_ids)
            results["token_type_ids"].append(token_type_ids)
            results["attention_mask"].append(attention_mask)
            results["raw_tokens"].append(raw_tokens)

        return results
