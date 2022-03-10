from archai.nlp.datasets_v2.tokenizer_utils.tokenizer_base import Tokenizer

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import WordLevelTrainer

class WordTokenizer(Tokenizer):
    """
    """

    def __init__(self, vocab_path: str) -> None:
        """
        """

        model = WordLevel(unk_token='UNK')
        trainer = WordLevelTrainer(special_tokens=['UNK'])

        super().__init__(model, trainer, vocab_path)
