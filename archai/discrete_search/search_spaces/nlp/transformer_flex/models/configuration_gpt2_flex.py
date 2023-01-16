# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config


def map_to_list(variable: Any, size: int) -> List[Any]:
    """Map variable to list of size.

    Args:
        variable: Variable to map to list.
        size: Size of list.

    Returns:
        List of `size` with variable mapped to it.

    """

    if isinstance(variable, list):
        size_diff = size - len(variable)

        if size_diff < 0:
            return variable[:size]
        elif size_diff == 0:
            return variable
        elif size_diff > 0:
            return variable + [variable[0]] * size_diff

    return [variable] * size


class GPT2FlexConfig(GPT2Config):
    model_type = "gpt2-flex"

    def __init__(self, *args, primer_square: Optional[bool] = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.primer_square = primer_square
        if primer_square:
            self.activation_function = "relu"

        self.n_inner = self.n_inner if self.n_inner is not None else 4 * self.n_embd
        self.n_inner = map_to_list(self.n_inner, self.n_layer)

        self.n_head = map_to_list(self.n_head, self.n_layer)
