# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based tasks classes."""

from archai.nlp.eval.harness.tasks.arc import (
    ARCChallengeHarnessTask,
    ARCEasyHarnessTask,
)
from archai.nlp.eval.harness.tasks.glue import (
    CoLAHarnessTask,
    MNLIMatchedHarnessTask,
    MNLIMismatchedHarnessTask,
    MRPCHarnessTask,
    QNLIHarnessTask,
    QQPHarnessTask,
    RTEHarnessTask,
    SST2HarnessTask,
    STSBHarnessTask,
    WNLIHarnessTask,
)
from archai.nlp.eval.harness.tasks.hella_swag import HellaSwagHarnessTask
from archai.nlp.eval.harness.tasks.human_eval import HumanEvalHarnessTask
from archai.nlp.eval.harness.tasks.lambada import (
    LambadaHarnessTask,
    LambadaOpenAIHarnessTask,
)
from archai.nlp.eval.harness.tasks.open_book_qa import OpenBookQAHarnessTask
from archai.nlp.eval.harness.tasks.piqa import PIQAHarnessTask
from archai.nlp.eval.harness.tasks.story_cloze import (
    StoryCloze2016HarnessTask,
    StoryCloze2018HarnessTask,
)
from archai.nlp.eval.harness.tasks.super_glue import (
    AXbHarnessTask,
    AXgHarnessTask,
    BoolQHarnessTask,
    CBHarnessTask,
    COPAHarnessTask,
    MultiRCHarnessTask,
    ReCoRDHarnessTask,
    WiCHarnessTask,
    WSCHarnessTask,
)
from archai.nlp.eval.harness.tasks.wino_grande import WinoGrandeHarnessTask
