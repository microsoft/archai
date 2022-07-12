"""
Script to play with featurizing a large dataset like The Pile, with the 
motivation to use downstream data summarization techniques on it.
"""

from datasets import load_dataset

from transformers import pipeline, AutoTokenizer, AutoModel

def main():
    checkpoint = "gpt2"
    model = AutoModel.from_pretrained(checkpoint, max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512, padding=True, truncation=True)
    embedder = pipeline("feature-extraction", model=model, tokenizer=tokenizer, model_max_length=512)
    sample_text = 'Good morning! How are you?'
    features = embedder(sample_text)
    
    # this outputs the same as features but in a nicer torch tensor format
    encoded_input = tokenizer(sample_text, return_tensors='pt')
    output = model(**encoded_input)
    features = output['last_hidden_state']

    print('done')

if __name__ == '__main__':
    main()