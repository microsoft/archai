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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # "facebook/opt-1.3b", "facebook/opt-350m" # "gpt2"
    checkpoint = "facebook/opt-350m" 
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    embedder = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    
    # load wikitext-103 validation and extract features on it 
    # using pipeline batching pipeline batching \
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    # dataset = load_dataset("wikitext", 'wikitext-103-v1', split='validation')
    dataset = load_dataset("the_pile", split='train')
    # max_length is not getting passed through properly
    # for idx, out in enumerate(embedder(KeyDataset(dataset, "text"), 
    #                     batch_size=100, 
    #                     truncation=True,
    #                     padding="max_length",
    #                     max_length=50, 
    #                     num_workers=20)):
    #     print(f'idx: {idx}, shape: {torch.tensor(out).shape}')
    
    # this outputs the same as features
    # and adds it to the dataset as an additional column
    # but it makes the dataset massive
    def featurize(examples):
        encoded_input = tokenizer(examples["text"], 
                        return_tensors='pt', 
                        padding=True,
                        truncation=True,
                        max_length=512).to(device)
        output = model(**encoded_input)
        return {"embedding" : output['last_hidden_state'].detach().cpu().numpy()}

    dataset = dataset.shard(21000, index=0)
    print(f"Sharded dataset length: {dataset.num_rows}")
    fdataset = dataset.map(featurize, batched=True, batch_size=42, num_proc=1)

    print('done')

if __name__ == '__main__':
    main()