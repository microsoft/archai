import os

class TokenizerFiles:
    def __init__(self, vocab_file:str, merges_file:str) -> None:
        self.vocab_file = vocab_file
        self.merges_file = merges_file

    @staticmethod
    def files_exists(save_dir:str)->bool:
        files = TokenizerFiles.from_path(save_dir)
        return os.path.exists(files.merges_file) and os.path.exists(files.vocab_file)

    @staticmethod
    def from_path(save_dir:str, save_prefix='tokenizer')->'TokenizerFiles':
        return TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                            merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))

