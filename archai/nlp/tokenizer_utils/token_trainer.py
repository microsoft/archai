from typing import Optional, Union, List
import logging
import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase, GPT2TokenizerFast, GPT2Tokenizer, PreTrainedTokenizer

from archai.common import utils, common
from archai.nlp.tokenizer_utils.tokenizer_files import TokenizerFiles
from archai.nlp.tokenizer_utils.token_config import TokenConfig


def train_tokenizer(lines:List[str], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 0, show_progress=False,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    # check if we already have tokenizer cached filed
    tokenizer_out_files = TokenizerFiles.from_path(save_dir=save_dir)
    if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
            and os.path.exists(tokenizer_out_files.merges_file):
        logging.info(f'Found BBPE tokenizer cached files at "{save_dir}", reusing them.')
        return tokenizer_out_files

    # TODO: measure impact of dropout
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    # function that will return us batches of lines
    def batch_iterator(batch_size=1000):
        for i in range(0, len(lines), batch_size):
            yield lines[i : i + batch_size]

    special_tokens = utils.dedup_list([stok for stok in                      \
        (token_config.unk_token, token_config.bos_token, token_config.eos_token,    \
            token_config.pad_token) \
        if stok])

    # train
    tokenizer.train_from_iterator(batch_iterator(),
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        show_progress=show_progress,
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=special_tokens)

    # additional tokens we might want to add
    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files

def create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig, max_length=1024)->PreTrainedTokenizerFast:
    tokenizer = GPT2TokenizerFast(vocab_file=tokenizer_files.vocab_file,
                              merges_file=tokenizer_files.merges_file,
                              model_max_length=max_length,
                              eos_token=token_config.eos_token,
                              bos_token=token_config.bos_token,
                              unk_token=token_config.unk_token,
                              pad_token=token_config.pad_token)

    # TODO: below shouldn't be required: https://github.com/huggingface/transformers/issues/664
    #tokenizer.padding_side = "left"
    return tokenizer
