from typing import Optional, Union, List
import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase, GPT2TokenizerFast

from archai.common import utils
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles

def train_tokenizer(lines:List[str], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 0, show_progress=False,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    tokenizer_out_files = TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                            merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))
    if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
            and os.path.exists(tokenizer_out_files.merges_file):
        return tokenizer_out_files

    # TODO: remove dropout
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(lines), batch_size):
            yield lines[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(),
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        show_progress=show_progress,
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files

def create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig, max_length:int)->PreTrainedTokenizerFast:
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

def main():
    token_config = TokenConfig()
    lines = [l["text"] for l in datasets["train"]]
    tokenizer_files = train_tokenizer(lines, token_config,
        show_progress=not training_args.disable_tqdm,
        vocab_size=transformer_args.vocab_size, save_dir=training_args.output_dir)
    logger.info("*** End Training Tokenizer***")
    tokenizer = create_tokenizer(tokenizer_files, token_config, transformer_args.max_length)
