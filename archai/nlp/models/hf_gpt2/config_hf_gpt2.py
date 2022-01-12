# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

from typing import Any, Dict

from archai.nlp.models.config_base import Config


class HfGPT2Config(Config):
    """Provides a configuration for HfGPT2.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the configuration.

        """

        super().__init__(**kwargs)

    @property
    def default(self) -> Dict[str, Any]:
        """Defines the default configuration used by the class.

        """

        return {}
