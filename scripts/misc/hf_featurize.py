"""
Script to play with featurizing a large dataset like The Pile, with the 
motivation to use downstream data summarization techniques on it.
"""
import torch

from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM


def main():

    checkpoint = "facebook/opt-1.3b" # "facebook/opt-350m" # "gpt2"
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # can we speed this up like we can do the dataset map way?
    # yes: see pipeline batching https://huggingface.co/docs/transformers/main_classes/pipelines
    embedder = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    sample_text = 'Good morning! How are you?'
    features = torch.tensor(embedder(sample_text))

    # load wikitext-103 validation 
    dataset = load_dataset("wikitext", 'wikitext-103-v1', split='validation')
    for idx, out in enumerate(embedder(KeyDataset(dataset, "text"), 
                        batch_size=100, 
                        truncation=None, 
                        num_workers=20)):
        print(f'idx: {idx}, shape: {torch.tensor(out).shape}')
    
    # this outputs the same as features
    # I know how to parallelize this via dataset.map
    encoded_input = tokenizer(sample_text, return_tensors='pt')
    output = model(**encoded_input)
    features = output['last_hidden_state']

    print('done')

if __name__ == '__main__':
    main()