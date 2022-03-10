from typing import Optional

#
SPECIAL_TOKENS = {
    'UNK': 0,
    'BOS': 1,
    'EOS': 2,
    'PAD': 3,
    'MASK': 4
}


class TokenConfig:
    """
    """

    def __init__(self,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False):
        """
        """

        #
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        #
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line
        self.lower_case = lower_case
