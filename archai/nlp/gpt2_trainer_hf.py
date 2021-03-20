import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List

# TODO: separate out torch dependencies from utils
if 'pydevd' in sys.modules:
    os.environ['vs_code_debugging'] = 'True'
def is_debugging()->bool:
    return 'vs_code_debugging' in os.environ and os.environ['vs_code_debugging']=='True'
if is_debugging():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets import load_dataset, DatasetDict, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed,
    GPT2TokenizerFast, GPT2Config,
    PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase
)
from tokenizers import ByteLevelBPETokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch

from archai.nlp.gpt_training_args import GptTrainingArguments

from archai.nlp.token_dataset import TokenConfig, TokenizerFiles
from archai.common import utils, common

logger = logging.getLogger(__name__)

"""
HF learning paramaeters (block_size==256, batch size==8) WikiText-2:
    - len(datasets['train']) == 36718 # number of examples == number of lines
    - lm_dataset['train'] == 8501 # number blocks
    - total steps: 2331
    - update steps/epoch == 1063 # number of blocks/batch size/grad_acc_steps
    - max_steps == total optimization steps == epochs * update steps/epoch == 3189
OpenAI:
    steps: 800K
    batch size: 512
    epochs: 60
    run time = 160hr on 128 TPU cores
    LR=6E-4 (GPT-3/125M, it reduces as model gets bigger)
    Train set: 26GB, 7B tokens
    Val set: 300MB, 900M tokens
"""

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class TransformerArguments:
    n_embd:int=768
    n_layer:int=12
    n_head:int=12
    max_length:int=1024
    vocab_size:int=50257
    gpt_config_name:str=''

known_gpt_configs = {
    'gpt2_small': TransformerArguments(gpt_config_name='gpt2_small'),
    'gpt2_medium': TransformerArguments(n_embd=1024, n_head=16, n_layer=24, gpt_config_name='gpt2_medium'),
    'gpt2_large': TransformerArguments(n_embd=1280, n_head=20, n_layer=36, gpt_config_name='gpt2_large'),
    'gpt2_xl': TransformerArguments(n_embd=1600, n_head=25, n_layer=48, gpt_config_name='gpt2_xl'),
    'gpt2_distill': TransformerArguments(n_layer=6, gpt_config_name='gpt2_distill'),
    'gpt2_tiny': TransformerArguments(n_embd=2, n_head=2, n_layer=2, gpt_config_name='gpt2_tiny'),
    'gpt2_toy': TransformerArguments(n_embd=2, n_head=2, n_layer=2, vocab_size=1000, max_length=32,
                                     gpt_config_name='gpt2_toy'),
    'gpt1': TransformerArguments(vocab_size=40478, max_length=512, gpt_config_name='gpt1'),
    'aitextgen': TransformerArguments(n_embd=256, n_head=8, n_layer=8, vocab_size=5000, max_length=32,
                                      gpt_config_name='aitextgen'),
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library), ex. 'wikitext'"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library), ex. 'wikitext-103-raw-v1'"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory to store downloaded dataset"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            #raise ValueError("Need either a dataset name or a training/validation file.")
            pass
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def get_checkpoint(output_dir:str, overwrite_output_dir:bool)->Optional[str]:
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use overwrite_output_dir=True"
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                 "Use overwrite_output_dir=True"
            )
    return last_checkpoint

def setup_logging(local_rank:int, expdir:str):
    utils.create_logger(filepath=os.path.join(expdir, 'logs.log'))

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

def dataset_from_files(train_file:Optional[str], validation_file:Optional[str])->DatasetDict:
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    extension = (
        train_file.split(".")[-1]
        if train_file is not None
        else validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    assert isinstance(datasets, DatasetDict)
    return datasets

def dataset_from_name(dataset_name:str, dataset_config_name:Optional[str],
                      data_dir:Optional[str], validation_split_percentage:Optional[int])->DatasetDict:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(dataset_name, dataset_config_name,
                            cache_dir=data_dir)
    assert isinstance(datasets, DatasetDict)

    if "validation" not in datasets.keys():
        datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=data_dir
        )
        datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=data_dir
        )

    return datasets

def model_from_pretrained(model_name_or_path:str, revision:str,
                          cache_dir:Optional[str], use_auth_token:Optional[bool])->PreTrainedModel:
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(model_name_or_path,
                cache_dir=cache_dir, revision=revision, use_auth_token=use_auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=revision,
        use_auth_token=use_auth_token,
    )
    return model

def model_from_config(model_config:PretrainedConfig)->PreTrainedModel:
    return AutoModelForCausalLM.from_config(model_config)

def tokenizer_from_pretrained(model_name_or_path:str, revision:str, use_fast:bool,
                          cache_dir:Optional[str], use_auth_token:Optional[bool])->PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name_or_path,
        cache_dir=cache_dir, revision=revision,
        use_auth_token=use_auth_token, use_fast=use_fast)

def create_lm_datasets(datasets:DatasetDict, tokenizer:PreTrainedTokenizerBase,
                       preprocessing_num_workers:Optional[int], overwrite_cache:bool,
                       block_size:Optional[int])->DatasetDict:
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # we assume that train/test split has same column names
    split = "train" if "train" in datasets else "test" if "test" in datasets else ""
    assert split != "", "dataset doesn't contain known split of train or test"
    column_names = datasets[split].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # bind function to column name
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
    )

    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples['input_ids']) # list(examples.keys())[0]
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )

    return lm_datasets

def get_datasets(data_args:DataTrainingArguments)->DatasetDict:
    if data_args.dataset_name is not None:
        datasets = dataset_from_name(data_args.dataset_name,
                                     data_args.dataset_config_name, data_args.data_dir,
                                     data_args.validation_split_percentage)
    elif data_args.train_file is not None:
        datasets = dataset_from_files(data_args.train_file, data_args.validation_file)
    else:
        raise ValueError('Either dataset_name or train_file must be provided')

    assert isinstance(datasets, DatasetDict)
    return datasets

def load_tokenizer(model_args:ModelArguments)->PreTrainedTokenizerBase:
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    assert tokenizer_name_or_path
    tokenizer = tokenizer_from_pretrained(tokenizer_name_or_path,
        model_args.model_revision, model_args.use_fast_tokenizer,
        model_args.cache_dir,
        True if model_args.use_auth_token else None)
    return tokenizer

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

def create_model(model_args:ModelArguments, input_embedding_size:int,
                 model_config:Optional[PretrainedConfig]=None)->PreTrainedModel:
    if model_args.model_name_or_path:
        model = model_from_pretrained(model_args.model_name_or_path,
                              model_args.model_revision,
                              model_args.cache_dir,
                              True if model_args.use_auth_token else None)
    elif model_config:
        model = model_from_config(model_config)
    else:
        raise ValueError('Either config_name or model_name_or_path or model_config must be provided')
    # if vocab size is not same as input token embedding size then resize input embedding
    model.resize_token_embeddings(input_embedding_size)

    return model

def train_model(checkpoint:Optional[str], trainer:Trainer,
               model_args:ModelArguments):

    if checkpoint is None and model_args.model_name_or_path is not None and \
            os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def evaluate_model(lm_datasets, trainer:Trainer):
    eval_dataset = lm_datasets['test'] if 'test' in lm_datasets else None # if none then use val set
    eval_output = trainer.evaluate(eval_dataset=eval_dataset)

    if 'perplexity' not in eval_output:
        perplexity = math.exp(eval_output["eval_loss"])
        eval_output["perplexity"] = perplexity

    trainer.log_metrics("eval", eval_output)
    trainer.save_metrics("eval", eval_output)

    return eval_output

def train_tokenizer(dataset:Dataset, token_config: TokenConfig,
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
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

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

def create_model_config(transformer_args:TransformerArguments,
        dropout=0.0, bos_token_id=0, eos_token_id=0)->PretrainedConfig:
    config = GPT2Config(vocab_size=transformer_args.vocab_size,
                        # n_ctx is dimensionality of the causal mask (usually same as n_positions).
                        n_ctx=transformer_args.max_length, n_positions=transformer_args.max_length,
                        n_embd=transformer_args.n_embd, n_layer=transformer_args.n_layer, n_head=transformer_args.n_head,
                        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                        resid_pdrop=dropout, embd_pdrop=dropout,
                        attn_pdrop=dropout, summary_first_dropout=dropout
                        )
    return config

def get_logdir(outdir:str) -> str:
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join(outdir, current_time + "_" + socket.gethostname())

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GptTrainingArguments, TransformerArguments),
                              description='GPT2 trainer')

    model_args, data_args, training_args, transformer_args = parser.parse_args_into_dataclasses()

    # create dataset and output dirs
    pt_data_dir, pt_output_dir = common.pt_dirs()
    data_args.data_dir = data_args.data_dir or pt_data_dir or common.default_dataroot()
    data_args.data_dir = utils.full_path(os.path.join(data_args.data_dir, 'textpred', 'huggingface', 'datasets'))
    training_args.output_dir =  utils.full_path(pt_output_dir or \
                    os.path.join(training_args.output_dir or '~/logdir', training_args.experiment_name)
                , create=True)
    training_args.logging_dir = get_logdir(training_args.output_dir) if training_args.logging_dir is None else training_args.logging_dir
    training_args.toy = utils.is_debugging() if training_args.toy is None else training_args.toy

    if transformer_args.gpt_config_name:
        transformer_args = known_gpt_configs[transformer_args.gpt_config_name]

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1.0e9
    gb3_blocks = int(gpu_mem/4) # 3gb blocks

    # if not specified, overrite is False if not toy mode
    training_args.overwrite_output_dir = training_args.overwrite_output_dir if \
        training_args.overwrite_output_dir is not None else training_args.toy
    if training_args.toy:
        # adjust for current GPU RAM on dev machine for batch size = 4
        transformer_args.max_length = min(256*gb3_blocks, transformer_args.max_length)

    data_args.dataset_name = data_args.dataset_name if data_args.dataset_name is not None else 'wikitext'
    if data_args.dataset_name == 'wikitext' and data_args.dataset_config_name is None:
        data_args.dataset_config_name = 'wikitext-2-raw-v1' if training_args.toy else 'wikitext-103-raw-v1'

    setup_logging(training_args.local_rank, training_args.output_dir)
    logging.info(f'toy={training_args.toy}, fp16={training_args.fp16}')
    logging.info(f'gpt_config_name={transformer_args.gpt_config_name}, n_embd={transformer_args.n_embd}, n_layer={transformer_args.n_layer}, n_embd={transformer_args.n_embd}, n_head={transformer_args.n_head}, max_length={transformer_args.max_length}, vocab_size={transformer_args.vocab_size}')
    logging.info(f'num_steps={training_args.max_steps}, epochs={training_args.num_train_epochs}')
    logging.info(f'n_gpus={training_args.n_gpu}, parallel_mode={training_args.parallel_mode}, device={training_args.device}, seed={training_args.seed}')
    logging.info(f'dataset={data_args.dataset_name}, dataset_config_name={data_args.dataset_config_name}, datadir="{data_args.data_dir}"')
    logging.info(f'expdir="{training_args.output_dir}"')
    logging.info(f'train_batch_size={training_args.per_device_train_batch_size}, fp16="{training_args.fp16}"')
    logging.info('')
    logging.info('')

    logger.info("transformer_args %s", transformer_args)
    logger.info("training_args %s", training_args)
    logger.info("data_args %s", data_args)
    logger.info("model_args %s", model_args)
    logging.info('')
    logging.info('')


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = get_datasets(data_args)

    token_config = TokenConfig()
    logger.info("*** Start Training Tokenizer***")
    tokenizer_files = train_tokenizer(datasets["train"], token_config,
        show_progress=not training_args.disable_tqdm,
        vocab_size=transformer_args.vocab_size, save_dir=training_args.output_dir)
    logger.info("*** End Training Tokenizer***")
    tokenizer = create_tokenizer(tokenizer_files, token_config, transformer_args.max_length)
    #tokenizer = load_tokenizer(model_args)

    lm_datasets = create_lm_datasets(datasets, tokenizer,
                                     data_args.preprocessing_num_workers,
                                     data_args.overwrite_cache, data_args.block_size)

    model_config = create_model_config(transformer_args)
    model = create_model(model_args, len(tokenizer), model_config=model_config)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        last_checkpoint = get_checkpoint(training_args.output_dir, training_args.overwrite_output_dir) if training_args.do_train else None

        logger.info("*** Training ***")
        train_model(last_checkpoint, trainer, model_args)
    if training_args.do_test:
        logger.info("*** Evaluate ***")
        evaluate_model(lm_datasets, trainer)

if __name__ == "__main__":
    main()
