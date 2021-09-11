import os

class DatasetFiles:
    def __init__(self, dataroot:str, dataset:str) -> None:
        self._known_datasets ={
            'wikitext-103': {'train_file': 'wiki.train.tokens',
                             'valid_file': 'wiki.valid.tokens',
                             'test_file': 'wiki.test.tokens'},
            'wikitext-2': {'train_file': 'wiki.train.tokens',
                             'valid_file': 'wiki.valid.tokens',
                             'test_file': 'wiki.test.tokens'}
        }

        dataset_paths = self._known_datasets[dataset]
        self.train_file = os.path.join(dataroot, dataset, dataset_paths['train_file'])
        self.valid_file = os.path.join(dataroot, dataset, dataset_paths['valid_file'])
        self.test_file = os.path.join(dataroot, dataset, dataset_paths['test_file'])