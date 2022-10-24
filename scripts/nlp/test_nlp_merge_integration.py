from archai.nlp.datasets.nvidia_datasets.hf_datasets.tokenizer_utils.gpt2_tokenizer import GPT2Tokenizer
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

tokenizer = GPT2Tokenizer()
tokenizer.train_from_iterator(dataset["train"])