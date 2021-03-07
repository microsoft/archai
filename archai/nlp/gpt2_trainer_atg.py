import argparse
from typing import Optional, Tuple, List, Union
import os
import logging
import textwrap

from pytorch_lightning import callbacks

import torch

from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import pytorch_lightning as pl

from archai.common import ml_utils, utils, common
from archai.nlp.token_dataset import DatasetFiles, TokenDataset, TokenizerFiles, TokenConfig
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

    tokenizer_out_files = TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                            merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))
    if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
            and os.path.exists(tokenizer_out_files.merges_file):
        return tokenizer_out_files

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)

    tokenizer.train(files=files,
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files


class GptConfig:
    def __init__(self, n_embd=768, n_layer=12, n_head=12, max_length=1024, vocab_size:int=50257) -> None:
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_length = max_length
        self.vocab_size = vocab_size

known_gpt_configs = {
    'gpt2_small': GptConfig(),
    'gpt2_medium': GptConfig(n_embd=1024, n_head=16, n_layer=24),
    'gpt2_large': GptConfig(n_embd=1280, n_head=20, n_layer=36),
    'gpt2_xl': GptConfig(n_embd=1600, n_head=25, n_layer=48),
    'gpt2_distill': GptConfig(n_layer=6),
    'gpt2_tiny': GptConfig(n_embd=2, n_head=2, n_layer=2),
    'gpt2_toy': GptConfig(n_embd=2, n_head=2, n_layer=2, vocab_size=1000, max_length=32),
    'gpt1': GptConfig(vocab_size=40478, max_length=512),
    'aitextgen': GptConfig(n_embd=256, n_head=8, n_layer=8, vocab_size=5000, max_length=32),
}

def create_model(gpt_config:GptConfig,
                 dropout=0.0, fp16=False,
                 bos_token_id=0, eos_token_id=0)->GPT2LMHeadModel:

    config = GPT2Config(vocab_size=gpt_config.vocab_size,
                        # n_ctx is dimensionality of the causal mask (usually same as n_positions).
                        n_ctx=gpt_config.max_length, n_positions=gpt_config.max_length,
                        n_embd=gpt_config.n_embd, n_layer=gpt_config.n_layer, n_head=gpt_config.n_head,
                        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                        resid_pdrop=dropout, embd_pdrop=dropout,
                        attn_pdrop=dropout, summary_first_dropout=dropout
                        )

    model = GPT2LMHeadModel(config=config)
    if fp16:
        model = model.half()

    return model

def create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig)->PreTrainedTokenizerFast:
    tokenizer = GPT2TokenizerFast(vocab_file=tokenizer_files.vocab_file,
                              merges_file=tokenizer_files.merges_file,
                              eos_token=token_config.eos_token,
                              bos_token=token_config.bos_token,
                              unk_token=token_config.unk_token,
                              pad_token=token_config.pad_token)
    tokenizer.padding_side = "left"

    return tokenizer

def train_model(dataset_files:DatasetFiles,
        model:PreTrainedModel, tokenizer: PreTrainedTokenizerFast,
        output_dir:str, num_steps = 5000, batch_size = 256,
        learning_rate = 1e-4, # reduce to 1e-3 if batch_size=1
        weight_decay = 0.05, adam_epsilon = 1e-8,
        max_grad_norm = 0.5, warmup_steps = 0, gradient_accumulation_steps = 1,
        save_every = 1000, generate_every = 1000,
        fp16: bool = False, fp16_opt_level: str = "O1",
        loggers: List = None, n_gpus:int=-1, # -1==use all GPUs
        num_workers:Optional[int] = None,
        benchmark: bool = True
    ) -> None:

        train_data = TokenDataset(train_file=dataset_files.train_file,
            tokenizer=tokenizer, block_size=model.config.n_positions)

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
            gpus=n_gpus,
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


def generate(model:PreTrainedModel, tokenizer:PreTrainedTokenizerFast, prompt:str,
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

    prompt_tensors = tokenizer(text)
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


def evaluate(eval_file:str, model:PreTrainedModel,
        tokenizer:PreTrainedTokenizerFast, max_eval_len=-1)->Tuple[float, float]: # returns avg segment loss and perplexity
    text = utils.read_string(eval_file)
    if max_eval_len >=0:
        text = text[:max_eval_len] if len(text) > max_eval_len else text

    # below will be just one giant tensor of all tokens
    encodings = tokenizer(text, return_tensors='pt')

    # max input segment length for the model
    max_length = model.config.n_positions
    # on avg, context length is half of segment length
    context_len = max_length // 2

    model.eval()

    lls = []
    with torch.no_grad():
        # we use sliding window algorithm so that at each step input segment is
        # i to i+max_len. We set target labels same as segment but ignore the
        # loss calculation for first context_len tokens. In next step,
        # i = i + context_len. This way we evaluate output label for each input
        # label except for the first context_len. Also note that huggingface model
        # shifts labels automatically by one which is why it is OK to set
        # targer_ids same as input_ids. Further, note that autorgressive model
        # can only see previous tokens in segment for each j-th token. This means
        # we are making prediction for each token as if someone typed sequentially
        # one after another in parallel in single forward call.
        for i in range(0, encodings.input_ids.size(1), context_len):
            begin_loc = max(i + context_len - max_length, 0)
            end_loc = min(i + context_len, encodings.input_ids.size(1))
            trg_len = end_loc - i    # may be different from context_len on last loop
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()

            # -100 is special number to indicate ignore in loss calculation
            # token ids are otherwise >=0
            target_ids[:,:-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            seg_loss = outputs[0] * trg_len # total loss = avg_loss per prediction * num_predictions

            lls.append(seg_loss)

        avg_seg_loss = torch.stack(lls).sum() / end_loc
        ppl = torch.exp(avg_seg_loss).item()

    return avg_seg_loss.item(), ppl


def main():
    parser = argparse.ArgumentParser(description='GPT2 trainer')
    parser.add_argument('--experiment-name', '-n', default='train_gpt2')
    parser.add_argument('--experiment-description', '-d', default='Train GPT2')

    parser.add_argument('--device', default='',
                        help='"cuda" or "cpu" or "" in which case use cuda if available')
    parser.add_argument('--n_gpus', type=int, default=-1)
    parser.add_argument('--train-batch-size', '-b', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--seed', '-s', type=float, default=42.0)
    parser.add_argument('--fp16', type=lambda x: x.lower() == 'true',
                        nargs='?', const=True, default=False)

    parser.add_argument('--datadir', default='',
                        help='where to find dataset files')
    parser.add_argument('--outdir', default='',
                        help='where to put results, default is ~/logdir')

    parser.add_argument('--dataset', default='wikitext-103')
    parser.add_argument('--toy',  type=lambda x: x.lower() == 'true',
                        nargs='?', const=True, default=utils.is_debugging(),
                        help='if true then override dataset and number of iterations to run quick sanity check if code compiles')
    parser.add_argument('--gpt-config-name', type=str, default='gpt2_small')
    parser.add_argument('--num-steps', type=int, default=5000)


    args = parser.parse_args()

    dataset:str = args.dataset
    num_steps:int = args.num_steps
    seed:float = args.seed
    fp16:bool = args.fp16
    train_batch_size:int = args.train_batch_size
    gpt_config_name = args.gpt_config_name
    toy:bool = args.toy
    max_eval_len = -1

    if toy:
        dataset = 'wikitext-2'
        num_steps = 1
        gpt_config_name = 'gpt2_toy'
        max_eval_len = 10000


    gpt_config = known_gpt_configs[gpt_config_name]

    # create dataset and output dirs
    pt_data_dir, pt_output_dir = common.pt_dirs()
    dataroot = args.datadir or pt_data_dir or common.default_dataroot()
    datadir = utils.full_path(os.path.join(dataroot, 'textpred'))
    expdir =  utils.full_path(pt_output_dir or \
                    os.path.join(args.outdir or '~/logdir', args.experiment_name)
                , create=True)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_gpus = 1 if utils.is_debugging() or utils.is_windows() else args.n_gpus

    utils.setup_cuda(seed)
    utils.create_logger(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(f'toy={toy}, num_steps={num_steps}, n_gpus={n_gpus}, seed={seed}')
    logging.info(f'dataset={dataset}, datadir="{datadir}"')
    logging.info(f'expdir="{expdir}"')
    logging.info(f'train_batch_size={train_batch_size}, fp16="{fp16}"')

    # paths for dataset
    dataset_files = DatasetFiles(datadir, dataset)

    # train tokenizer
    token_config = TokenConfig()
    tokenizer_files = train_tokenizer(files=dataset_files.train_file,
        token_config=token_config, vocab_size=gpt_config.vocab_size, save_dir=expdir)

    # load tokenizer from trained files
    tokenizer = create_tokenizer(tokenizer_files, token_config)

    # create model
    model = create_model(gpt_config=gpt_config, fp16=fp16)

    logging.info(f'model_size={ml_utils.param_size(model)/1.0E6}M')

    logging.info('training...')
    train_model(dataset_files, model, tokenizer, expdir, n_gpus=n_gpus,
                num_steps=num_steps, batch_size=train_batch_size)

    logging.info(f'GPU alloc mem: {torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]/1.0E6}MB')
    logging.info(f'GPU resv mem: {torch.cuda.memory_stats(device)["reserved_bytes.all.peak"]/1.0E6}MB')

    logging.info('evaluating...')
    model.to('cpu')
    ml_utils.clear_gpu_memory()
    model.to(device)
    loss, ppl = evaluate(dataset_files.test_file, model, tokenizer,
                         max_eval_len=max_eval_len)

    logging.info(f'GPU alloc mem: {torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]/1.0E6}MB')
    logging.info(f'GPU resv mem: {torch.cuda.memory_stats(device)["reserved_bytes.all.peak"]/1.0E6}MB')

    print('loss:', loss)
    print('ppl:', ppl)

if __name__ == '__main__':
    main()