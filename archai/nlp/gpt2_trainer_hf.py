import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List

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
    TrainingArguments,
    default_data_collator,
    set_seed,
    PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase
)
from tokenizers import ByteLevelBPETokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from archai.nlp.token_dataset import TokenConfig, TokenizerFiles
from archai.common import utils

logger = logging.getLogger(__name__)


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
            raise ValueError("Need either a dataset name or a training/validation file.")
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

def setup_logging(local_rank:int):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(local_rank) else logging.WARN)

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
                            data_dir=data_dir)
    assert isinstance(datasets, DatasetDict)

    if "validation" not in datasets.keys():
        datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
            data_dir=data_dir
        )
        datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
            data_dir=data_dir
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

def create_lm_datasets(do_train:bool, datasets:DatasetDict, tokenizer:PreTrainedTokenizerBase,
                       preprocessing_num_workers:Optional[int], overwrite_cache:bool,
                       block_size:Optional[int])->Dataset:
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
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
        total_length = len(concatenated_examples[list(examples.keys())[0]])
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

    assert datasets is DatasetDict
    return datasets

def create_tokenizer(model_args:ModelArguments)->PreTrainedTokenizerBase:
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    assert tokenizer_name_or_path
    tokenizer = tokenizer_from_pretrained(tokenizer_name_or_path,
        model_args.model_revision, model_args.use_fast_tokenizer,
        model_args.cache_dir,
        True if model_args.use_auth_token else None)
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

def train_model(lm_datasets, model:PreTrainedModel, tokenizer:PreTrainedTokenizerBase,
               training_args:TrainingArguments,
               model_args:ModelArguments):
    # Detecting last checkpoint.
    last_checkpoint = get_checkpoint(training_args.output_dir, training_args.overwrite_output_dir) if training_args.do_train else None

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

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        evaluate(lm_datasets, trainer)

def evaluate(lm_datasets, trainer:Trainer):
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
                    dropout: float = None, min_frequency: int = 2,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    tokenizer_out_files = TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                            merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))
    if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
            and os.path.exists(tokenizer_out_files.merges_file):
        return tokenizer_out_files

    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator,
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments),
                              description='GPT2 trainer')


    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args.local_rank)
    logger.info("Training/evaluation parameters %s", training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = get_datasets(data_args)

    tokenizer = create_tokenizer(model_args)

    lm_datasets = create_lm_datasets(training_args.do_train, datasets, tokenizer,
                                     data_args.preprocessing_num_workers,
                                     data_args.overwrite_cache, data_args.block_size)

    model = create_model(model_args, len(tokenizer), model_config)

    train_main(lm_datasets, model, tokenizer, training_args, model_args)


if __name__ == "__main__":
    main()
