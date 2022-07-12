"""
Script to play with featurizing a large dataset like The Pile, with the 
motivation to use downstream data summarization techniques on it.
"""

from datasets import load_dataset


def main():

    # load wikitext-103
    dataset = load_dataset("wikitext", 'wikitext-103-v1')

    # grab checkpoint and tokenizer
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    # tokenization method 1
    # mimics https://huggingface.co/course/chapter3/2?fw=pt
    # this is bad because it actually doesn't return a dataset
    # but a dictionary with 'input_ids' as the key and is slow
    # tokenized_dataset = tokenizer(dataset['train']['text'])
    
    # tokenize method 2
    # mimics https://huggingface.co/course/chapter3/2?fw=pt
    # and is the recommended way of tokenizing a dataset
    # as it returns an actual dataset with the 
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_dataset = dataset.map(tokenize_function, split='validation', 
                                    batched=True, batch_size=20000, num_proc=48)
    tokenized_dataset.save_to_disk('tokenized-wt103-v1')

    print("done")

    


if __name__ == '__main__':
    main()