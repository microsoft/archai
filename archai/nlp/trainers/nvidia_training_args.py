# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from dataclasses import dataclass, field


@dataclass
class NvidiaTrainingArguments:
    """
    """

    experiment_name: str = field(metadata={"help": "Name of the experiment."})

    experiment_desc: str = field(default="", metadata={"help": "Description of the experiment."})

