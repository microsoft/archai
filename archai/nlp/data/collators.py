# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Collator-related classes and methods to pre-process data prior to the model.
"""

import importlib

from transformers.data.data_collator import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)

from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)

# Available data collators
DATA_COLLATORS = {
    "language_modelling": "LanguageModelingCollator",
    "privacy_language_modelling": "PrivacyLanguageModelingCollator",
    "seq2seq": "Seq2SeqCollator",
    "sequence_classification": "SequenceClassificationCollator",
    "token_classification": "TokenClassificationCollator",
}


def load_collator(collator_name: str, **kwargs) -> DataCollator:
    """Instantiates a new collator.

    Args:
        collator_name: Name of data collator to be instantiated.

    Returns:
        (DataCollator): A data collator wrapped into corresponding class.

    """

    collator_name = collator_name.replace("-", "_")
    if collator_name in DATA_COLLATORS.keys():
        collator_cls_name = DATA_COLLATORS[collator_name]
    else:
        raise NotImplementedError(f"collator: {collator_name} has not been implemented yet.")

    collator_module = importlib.import_module("archai_nlp.data.collators")
    collator_cls = getattr(collator_module, collator_cls_name)

    logger.info(f"Loading data collator: {collator_name}")

    collator = collator_cls(**kwargs)

    logger.info("Data collator loaded.")

    return collator


class LanguageModelingCollator(DataCollatorForLanguageModeling):
    """Data collator used for causal and masked language modeling tasks."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the standard language modeling collator."""

        super().__init__(*args, **kwargs)


class Seq2SeqCollator(DataCollatorForSeq2Seq):
    """Data collator used for sequence-to-sequence tasks."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the standard sequence-to-sequence collator."""

        super().__init__(*args, **kwargs)


class SequenceClassificationCollator(DataCollatorWithPadding):
    """Data collator used for sequence classification tasks."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the standard sequence classification collator."""

        super().__init__(*args, **kwargs)


class TokenClassificationCollator(DataCollatorForTokenClassification):
    """Data collator used for token classification tasks."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the standard token classification collator."""

        super().__init__(*args, **kwargs)
