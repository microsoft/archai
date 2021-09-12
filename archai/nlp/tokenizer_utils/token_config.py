from typing import List, Optional
from archai.common import utils

class TokenConfig:
    def __init__(self, bos_token:Optional[str]="<|endoftext|>", eos_token:Optional[str]="<|endoftext|>",
                 unk_token:Optional[str]="<|endoftext|>", pad_token:Optional[str]=None,
                 add_prefix_space=False, add_prefix_new_line=False) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line

    def get_special_tokens(self)->List[str]:
        return utils.dedup_list([stok for stok in                      \
        (self.unk_token, self.bos_token, self.eos_token,    \
            self.pad_token) \
        if stok])