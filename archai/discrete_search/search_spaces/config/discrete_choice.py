# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from random import Random
from numbers import Number
from typing import Any, List, Union, Optional


class DiscreteChoice:
    def __init__(self, choices: List[Union[int, float, str]],
                 probabilities: Optional[List[float]] = None, 
                 encode_strategy: str = 'auto') -> None:
        """ Stores a discrete choice of numeric or non-numeric values.
        The choice can be encoded as a numeric value or using one-hot encoding depending on the
        value passed to `encode_strategy`.

        Args:
            choices (List[Union[int, float, str]]): List of choices. Choices can be integers, 
                floats or strings.
            probabilities (Optional[List[float]], optional): Probability distribution of each choice
                used during sampling. If `None`, a uniform distribution is used.
            encode_strategy (str, optional): Encoding strategy to use ['one_hot', 'numeric']. If
                'auto', the encoding strategy is chosen based on the type of the choices.
                Defaults to 'auto'.
        """

        self.choices = choices
        self.probabilities = probabilities

        if encode_strategy == 'auto':
            encode_strategy = (
                'numeric' if all(isinstance(choice, Number) for choice in choices) 
                else 'one_hot'
            )

        self.encode_strategy = encode_strategy

    def __getitem__(self, idx: str) -> Any:
        return self.choices[idx]

    def __repr__(self) -> str:
        return f"DiscreteChoice({repr(self.choices)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.choices)

    def encode(self, option: Any) -> List[float]:
        """Encodes the option into a numeric value or a one-hot encoding.

        Args:
            option (Any): Option to encode.

        Returns:
            List[float]: Encoded option.
        """

        if self.encode_strategy == 'one_hot':
            assert option in self.choices, f'Invalid option: {option}. Valid options: {self.choices}'
            return [float(choice == option) for choice in self.choices]
        
        return [float(option)]

    def random_sample(self, rng: Optional[Random] = None) -> Any:
        """Randomly samples a choice from the discrete set. 

        Args:
            rng (Optional[Random], optional): Random number generator.

        Returns:
            Any: Randomly sampled choice.
        """
        rng = rng or Random()
        return rng.choices(self.choices, weights=self.probabilities, k=1)[0]
