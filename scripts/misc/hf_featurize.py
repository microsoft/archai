"""
Script to play with featurizing a large dataset like The Pile, with the 
motivation to use downstream data summarization techniques on it.
"""
from base64 import encode
from math import trunc
import torch

from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM


def main():

    checkpoint = "facebook/opt-1.3b" # "facebook/opt-350m" # "gpt2"
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    embedder = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    
    # load wikitext-103 validation and extract features on it 
    # using pipeline batching pipeline batching \
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    dataset = load_dataset("wikitext", 'wikitext-103-v1', split='validation')
    # for idx, out in enumerate(embedder(KeyDataset(dataset, "text"), 
    #                     batch_size=100, 
    #                     truncation=None, 
    #                     num_workers=20)):
    #     print(f'idx: {idx}, shape: {torch.tensor(out).shape}')
    
    # this outputs the same as features
    # and adds it to the dataset as an additional column
    # but it makes the dataset massive
    def featurize(examples):
        encoded_input = tokenizer(examples["text"], 
                        return_tensors='pt', 
                        padding=True,
                        truncation=True)
        output = model(**encoded_input)
        return {"embedding" : output['last_hidden_state']}

    dataset.map(featurize, batched=True, batch_size=1000)

    print('done')

if __name__ == '__main__':
    main()