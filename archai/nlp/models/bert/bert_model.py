# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""BERT-based transformer.
"""

from transformers.models.bert.configuration_bert import BertConfig as BertCfg
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForPreTraining,
    BertForSequenceClassification,
    BertLMHeadModel,
)

from archai.nlp.model import ArchaiModel


class BERTConfig(BertCfg):
    """Wraps a BERT transformer configuration."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the configuration of the transformer."""

        super().__init__(*args, **kwargs)


class BERTForCLM(BertLMHeadModel, ArchaiModel):
    """Wraps a BERT transformer for causal language modeling."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)


class BERTForMLM(BertForMaskedLM, ArchaiModel):
    """Wraps a BERT transformer for masked language modeling."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)


class BERTForSequenceClassification(BertForSequenceClassification, ArchaiModel):
    """Wraps a BERT transformer for sequence classification."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)


class BERTForPreTraining(BertForPreTraining, ArchaiModel):
    """Wraps a BERT transformer for pre-training (next sentence prediction and masked language modeling)."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)
