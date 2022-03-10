from archai.nlp.datasets_v2.dataset_loader import load_file_dataset, load_hub_dataset
from archai.nlp.datasets_v2.corpus import get_corpus


# dataset = load_file_dataset('/home/gderosa/dataroot/textpred/olx_WordData20211003',
#                             '/home/gderosa/dataroot/textpred/olx_WordData20211003/cache',
#                             data_files=['test.txt'])

# print(dataset)

# hub_dataset = load_hub_dataset('wikitext', 'wikitext-103-v1', '/home/gderosa/dataroot/textpred/wt103/cache')
# print(dataset)
# dataset = dataset.map(lambda x: {"_text": x["text"]}, batched=True)
# print(dataset)

corpus = get_corpus('/home/gderosa/dataroot/textpred/olx_WordData20211003',
                    '/home/gderosa/dataroot/textpred/olx_WordData20211003/cache',
                    data_files={'train': 'valid.txt'},
                    refresh_cache=False)

print(corpus.vocab)
print(corpus.dataset)