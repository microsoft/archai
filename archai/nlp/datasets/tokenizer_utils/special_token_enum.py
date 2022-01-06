from enum import Enum


class SpecialTokenEnum(Enum):
    UNK=0
    BOS=1
    EOS=2
    PAD=3
    MASK=4