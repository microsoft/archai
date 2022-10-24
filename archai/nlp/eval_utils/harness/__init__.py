# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizes a harness evaluation tool based on EleutherAI/lm-evaluation-harness.
"""

from archai.nlp.eval_utils.harness.harness_eval import evaluate
from archai.nlp.eval_utils.harness.harness_model import HarnessModel
from archai.nlp.eval_utils.harness.harness_task import load_harness_task
from archai.nlp.eval_utils.harness.harness_utils import (
    HarnessCall,
    HarnessCallFactory,
    MultipleTokenStoppingCriteria,
    clean_sample_text,
)
