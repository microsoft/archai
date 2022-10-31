from typing import List, Any

class DiscreteChoice():
    def __init__(self, choices: List[Any]):
        self.choices = choices

    def __getitem__(self, idx):
        return self.choices[idx]

    def __repr__(self):
        return f'DiscreteChoice({repr(self.choices)})'
    
    def __str__(self):
        return self.__repr__()
    
    def get_search_params(self, blank: bool = False):
        if blank:
            return False
        return self

