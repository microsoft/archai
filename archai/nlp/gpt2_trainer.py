import argparse
from typing import Optional, Tuple, List, Union
import os
import logging
from pytorch_lightning import callbacks

import torch
from torch import nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoConfig
from tokenizers import ByteLevelBPETokenizer
import pytorch_lightning as pl

from archai.common import ml_utils, utils, common
from archai.nlp.token_dataset import TokenDataset, TokenizerFiles, TokenConfig
from archai.nlp.transformer_lightning import TransformerLightning
from archai.nlp.trainer_callback import TrainerCallback


def train_tokenizer(files: Union[str, List[str]], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 2,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    assert isinstance(files, str) or isinstance(files, list ), "files must be a string or a list."
    assert isinstance(added_tokens, list), "added_tokens must be a list."

    if isinstance(files, str):
        files = [files]

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)

    tokenizer.train(files=files, vocab_size=vocab_size-len(added_tokens),
        min_frequency=min_frequency,
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                          merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))


def create_models(tokenizer_files:TokenizerFiles, token_config: TokenConfig,
                 vocab_size:int, n_embd=256, n_layer=8, n_head=8,

                 max_length=32, # max seq length
                 dropout=0.0, fp16=False,
                 bos_token_id=0, eos_token_id=0)->Tuple[GPT2LMHeadModel, GPT2Tokenizer]:

    config = GPT2Config(vocab_size=vocab_size,
                        # n_ctx is dimensionality of the causal mask (usually same as n_positions).
                        n_ctx=max_length, n_positions=max_length,
                        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                        resid_pdrop=dropout, embd_pdrop=dropout,
                        attn_pdrop=dropout, summary_first_dropout=dropout
                        )

    model = GPT2LMHeadModel(config=config)
    if fp16:
        model = model.half()

    tokenizer = GPT2Tokenizer(vocab_file=tokenizer_files.vocab_file,
                              merges_file=tokenizer_files.merges_file,
                              eos_token=token_config.eos_token,
                              bos_token=token_config.bos_token,
                              unk_token=token_config.unk_token,
                              pad_token=token_config.pad_token)
    tokenizer.padding_side = "left"

    return model, tokenizer


def train_model(train_file:str, tokenizer_files:TokenizerFiles, token_config: TokenConfig,
        model:GPT2LMHeadModel, output_dir:str,
        num_steps = 5000, batch_size = 256,
        learning_rate = 1e-4, # reduce to 1e-3 if batch_size=1
        weight_decay = 0.05, adam_epsilon = 1e-8,
        max_grad_norm = 0.5, warmup_steps = 0, gradient_accumulation_steps = 1,
        save_every = 1000, generate_every = 1000,
        fp16: bool = False, fp16_opt_level: str = "O1",
        n_generate = 1, loggers: List = None,
        num_workers:Optional[int] = None, line_by_line=False,
        benchmark: bool = True, avg_loss_smoothing = 0.01
    ) -> None:

        train_data = TokenDataset(train_file=train_file, tokenizer_files=tokenizer_files,
                                  token_config=token_config,
                                  block_size=model.config.n_positions)

        if num_workers is None:
            num_workers = os.cpu_count() * 2 if not utils.is_debugging() else 0

        hparams = dict(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            num_steps=num_steps,
            pin_memory=True,
            num_workers=num_workers,
            save_every=save_every,
            generate_every=generate_every,
        )

        # Wrap the model in a pytorch-lightning module
        model = model.train()
        transformer_lightning = TransformerLightning(model, train_data, hparams)

        train_params = dict(
            #accelerator='ddp',
            accumulate_grad_batches=gradient_accumulation_steps,
            gpus=1 if utils.is_debugging() else -1, # use all GPUs
            max_steps=num_steps,
            gradient_clip_val=max_grad_norm if not fp16 else 0,
            checkpoint_callback=False,
            logger=loggers if loggers else False,
            weights_summary=None,
            callbacks=[TrainerCallback()]
        )

        if fp16:
            train_params["precision"] = 16 if fp16 else 32
            train_params["amp_level"] = fp16_opt_level

        if benchmark:
            train_params["benchmark"] = True

        trainer = pl.Trainer(**train_params)
        trainer.fit(transformer_lightning)

        model.save_pretrained(output_dir)

def generate(model, tokenizer, prompt:str,
             min_length: int = None, max_length: int = 256,
            temperature: float = 0.7, do_sample: bool = True,
            top_k:Optional[int] = None, top_p: Optional[float] = None,
            num_return_sequences=1) -> List[str]:
    """
    Generates texts using the Transformers model.
    Currently generates text using the model's generate() function.

    :param max_length: Maximum length for the generated text
    :param top_k: Limits the generated guesses to the top k guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set top_k=40)
    :param top_p: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with top_p=0.9)
    :param temperature: Determines the "creativity" of the generated text.
    The value range is different for each type of Transformer.
    :param do_sample: Samples the text, which is what we want. If False,
    the generated text will be the optimal prediction at each time,
    and therefore deterministic.
    """

    # if prompt:
    #     assert (
    #         len(prompt) < model.config.n_positions
    #     ), "The prompt is too large for the model."

    prompt_tensors = tokenizer(text=prompt, return_tensors="pt")
    input_ids = prompt_tensors["input_ids"].to(model.device)

    pad_token_id = tokenizer.pad_token_id

    # prevent an error from using a length greater than the model
    max_length = min(model.config.n_positions, max_length)

    outputs = model.generate(input_ids=input_ids, top_k=top_k, top_p=top_p,
        min_length=min_length, max_length=max_length,
        temperature=temperature, do_sample=do_sample,
        num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)

    if num_return_sequences > 1:
        gen_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    else:
        gen_texts = [tokenizer.decode(outputs[0], skip_special_tokens=True)]


    return gen_texts

def main():
    parser = argparse.ArgumentParser(description='GPT2 trainer')
    parser.add_argument('--experiment-name', '-n', default='train_gpt2')
    parser.add_argument('--experiment-description', '-d', default='Train GPT2')
    parser.add_argument('--epochs', '-e', type=int, default=108)
    parser.add_argument('--device', default='',
                        help='"cuda" or "cpu" or "" in which case use cuda if available')
    parser.add_argument('--train-batch-size', '-b', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--seed', '-s', type=float, default=42)
    parser.add_argument('--half', type=lambda x: x.lower() == 'true',
                        nargs='?', const=True, default=False)

    parser.add_argument('--datadir', default='',
                        help='where to find dataset files, default is ~/torchvision_data_dir')
    parser.add_argument('--outdir', default='',
                        help='where to put results, default is ~/logdir')

    parser.add_argument('--train-file', default='wiki.train.tokens', # 'tiny_shakespeare.txt'
                        help='training text file')
    parser.add_argument('--vocab-size', type=int, default=5000)
    parser.add_argument('--num-steps', type=int, default=5000 if not utils.is_debugging() else 1)

    args = parser.parse_args()

    if not args.datadir:
        args.datadir = common.default_dataroot()
    if not args.outdir:
        args.outdir = os.environ.get('PT_OUTPUT_DIR', '')
        if not args.outdir:
            args.outdir = os.path.join('~/logdir', 'gpt2_trainer', args.experiment_name)

    expdir = utils.full_path(args.outdir)
    os.makedirs(expdir, exist_ok=True)
    outdir = utils.full_path(args.outdir)
    datadir = utils.full_path(args.datadir)

    utils.setup_cuda(args.seed)

    utils.create_logger(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(
        f'exp_name="{args.experiment_name}", exp_desc="{args.experiment_description}"')
    logging.info('seed={args.seed}, epochs={args.epochs}, half={args.half}')
    logging.info(f'datadir="{datadir}"')
    logging.info(f'expdir="{expdir}"')
    logging.info(f'train_batch_size={args.train_batch_size}')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_config = TokenConfig()
    train_file = os.path.join(datadir, 'textpred', 'wikitext-103', args.train_file)
    num_steps = args.num_steps
    vocab_size:int = args.vocab_size
    tokenizer_files = train_tokenizer(files=train_file, token_config=token_config,
                    vocab_size=vocab_size, save_dir=outdir)

    model, tokenizer = create_models(tokenizer_files, token_config, vocab_size)
    model.to(device)
    train_model(train_file, tokenizer_files, token_config, model, outdir,
                num_steps=num_steps)

    print(generate(model, tokenizer, 'I wanted to write you this email because'))

if __name__ == '__main__':
    main()