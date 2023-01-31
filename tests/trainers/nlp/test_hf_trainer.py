# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile

import torch
from transformers import TrainerState, TrainingArguments

from archai.trainers.nlp.hf_trainer import HfTrainer


def test_hf_trainer_rotate_checkpoints():
    model = torch.nn.Linear(10, 5)
    args = TrainingArguments("tmp", save_total_limit=2, load_best_model_at_end=False)
    trainer = HfTrainer(model, args=args)

    state = TrainerState(best_model_checkpoint=None)
    trainer.state = state

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_1 = os.path.join(temp_dir, "checkpoint-1")
        os.mkdir(checkpoint_1)
        checkpoint_2 = os.path.join(temp_dir, "checkpoint-2")
        os.mkdir(checkpoint_2)
        checkpoint_3 = os.path.join(temp_dir, "checkpoint-3")
        os.mkdir(checkpoint_3)

        # Assert that nothing happens when `save_total_limit` is None or 0
        trainer.args.save_total_limit = None
        trainer._rotate_checkpoints(output_dir=temp_dir)
        assert os.path.exists(checkpoint_1)
        assert os.path.exists(checkpoint_2)
        assert os.path.exists(checkpoint_3)

        trainer.args.save_total_limit = 0
        trainer._rotate_checkpoints(output_dir=temp_dir)
        assert os.path.exists(checkpoint_1)
        assert os.path.exists(checkpoint_2)
        assert os.path.exists(checkpoint_3)

        # Assert that only the oldest checkpoint is deleted
        trainer.args.save_total_limit = 2
        trainer._rotate_checkpoints(output_dir=temp_dir)
        assert not os.path.exists(checkpoint_1)
        assert os.path.exists(checkpoint_2)
        assert os.path.exists(checkpoint_3)

        # Assert that the last checkpoint is not deleted when `load_best_model_at_end` is True
        trainer.args.load_best_model_at_end = True
        trainer.state.best_model_checkpoint = checkpoint_3
        trainer._rotate_checkpoints(output_dir=temp_dir)
        assert not os.path.exists(checkpoint_1)
        assert os.path.exists(checkpoint_2)
        assert os.path.exists(checkpoint_3)
