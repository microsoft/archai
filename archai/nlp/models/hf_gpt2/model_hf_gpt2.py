# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

import types
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from archai.nlp.models.hf_gpt2.config_hf_gpt2 import HfGPT2Config, HfGPT2FlexConfig
from archai.nlp.models.hf_gpt2.hf_gpt2_utils.gpt2_lm_head_model_flex import GPT2LMHeadModelFlex
from archai.nlp.models.model_base import ArchaiModel


class HfGPT2(ArchaiModel):
    """Huggingface's Open AI GPT-2 standard architecture.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.

        """

        super().__init__()

        self.config = HfGPT2Config(**kwargs)
        self.model = GPT2LMHeadModel(self.config)

        if self.config.tie_weight:
            self.model.tie_weights()

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:
<<<<<<< HEAD
        """Performs forward pass over the model.

        Args:
            input_ids: Input tokens.
            labels: Input labels (same as tokens).
            mems: Memory tensor.
            past_key_values: Tensor with past key/values.
            output_loss: Whether loss should be outputted.
            output_prediction_scores: Whether prediction scores should be outputted.

        Returns:
            (Tuple[torch.Tensor, ...]): Outputs, such as loss, prediction scores,
                memories and past key/values.

<<<<<<< HEAD
        """
=======
    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        # There is no memory in GPT-2
        pass
=======
        assert mems is None, 'HfGPT2 does not support memory (mems).'

        # Labels are the same as input_ids because they will be shifted inside the model
        # Causal attention mask is also created inside the model
        outputs = self.model(input_ids=input_ids,
                             labels=input_ids,
                             attention_mask=torch.ones_like(input_ids),
                             past_key_values=past_key_values)

        if output_loss:
            return (outputs.loss, None, None, outputs.past_key_values)
        
        if output_prediction_scores:
            # GPT-2 only outputs the logits, so they need to be converted with log_softmax
            return (None, F.log_softmax(outputs.logits, dim=-1), None, outputs.past_key_values)
>>>>>>> 0a1d1a35 (chore(hf_gpt2): Re-structures hf_gpt2-related files.)

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['embedding'] = self.get_params_from_layer(['Embedding'])
        params['attention'] = self.get_params_from_layer(['GPT2Attention'])
        params['ff'] = self.get_params_from_layer(['GPT2MLP'])
        params['layer_norm'] = self.get_params_from_layer(['LayerNorm'])

        params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
        params['total'] = params['non_embedding'] + params['embedding']
>>>>>>> 628e74a0 (fix(hf_gpt2): Fixes parameters calculation for HfGPT2 and HfGPT2Flex.)

<<<<<<< HEAD
        assert mems is None, 'GPT2 does not support memory (mems)'

        # Labels in Huggingface's GPT-2 are the same as inputs_ids and they will be shifted inside the model
        # Causal attention mask is created inside the model
        hf_out = self.model(input_ids=input_ids,
                            labels=input_ids,
                            attention_mask=torch.ones_like(input_ids))
=======
        return params


class HfGPT2Flex(ArchaiModel):
    """Huggingface's Open AI GPT-2 flex-based architecture.

    Flex-based architectures allow different hyperparameters settings for each layer.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.
        
        """
        
        super().__init__()

        self.config = HfGPT2FlexConfig(**kwargs)
        assert all(self.config.n_head[0] == n_h for n_h in self.config.n_head), 'HfGPT2Flex does not support different `n_head`.'

        self.model = GPT2LMHeadModelFlex(self.config)

        if self.config.tie_weight:
            self.model.tie_weights()

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:
        assert mems is None, 'HfGPT2Flex does not support memory (mems).'
>>>>>>> 0a1d1a35 (chore(hf_gpt2): Re-structures hf_gpt2-related files.)

        # Labels are the same as input_ids because they will be shifted inside the model
        # Causal attention mask is also created inside the model
        outputs = self.model(input_ids=input_ids,
                             labels=input_ids,
                             attention_mask=torch.ones_like(input_ids),
                             past_key_values=past_key_values)

<<<<<<< HEAD
    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        """Resets the length of the memory.

        Args:
            tgt_len: Length of target sample.
            ext_len: Length of extended memory.
            mem_len: Length of the memory.

        """

        # There is no memory in GPT-2
        pass
=======
        if output_loss:
            return (outputs.loss, None, None, outputs.past_key_values)
        
        if output_prediction_scores:
            # GPT-2 only outputs the logits, so they need to be converted with log_softmax
            return (None, F.log_softmax(outputs.logits, dim=-1), None, outputs.past_key_values)
>>>>>>> 0a1d1a35 (chore(hf_gpt2): Re-structures hf_gpt2-related files.)

    def get_params(self) -> Dict[str, int]:
        """Returns a dictionary of total parameters per implemented layer.

        Returns:
            (Dict[str, int]): Number of total parameters.

        """

        params = {}

        params['embedding'] = self.get_params_from_layer(['Embedding'])
        params['attention'] = self.get_params_from_layer(['GPT2Attention'])
        params['ff'] = self.get_params_from_layer(['GPT2MLPFlex'])
        params['layer_norm'] = self.get_params_from_layer(['LayerNorm'])

        params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
        params['total'] = params['non_embedding'] + params['embedding']

        return params
