from collections.abc import Sized

def map_to_list(p, n):
    if isinstance(p, Sized):
        if len(p) == 1:
            return p * n
        return p
    return [p] * n
