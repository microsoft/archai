# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archai.trainers.nlp.nvidia_trainer import save_checkpoint


def test_save_checkpoint():
    output_dir = tempfile.mkdtemp()
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.cuda.amp.GradScaler()
    trainer_state = {"step": 0}

    # Assert that the checkpoint file exists
    save_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state=trainer_state,
        fp16=False,
        save_all_checkpoints=False,
        is_best_model=False,
    )
    checkpoint_path = os.path.join(output_dir, "checkpoint-last.pt")
    assert os.path.exists(checkpoint_path)

    # Assert that the checkpoint file contains the expected data
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["model_config"] == model.config
    for key in checkpoint["model_state"]:
        assert torch.equal(checkpoint["model_state"][key], model.state_dict()[key])
    for key in checkpoint["optimizer_state"]:
        assert checkpoint["optimizer_state"][key] == optimizer.state_dict()[key]
    for key in checkpoint["scheduler_state"]:
        assert checkpoint["scheduler_state"][key] == scheduler.state_dict()[key]
    assert checkpoint["scaler_state"] is None
    assert checkpoint["trainer_state"] == trainer_state

    # Assert that the best model checkpoint file exists
    save_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state=trainer_state,
        fp16=False,
        save_all_checkpoints=False,
        is_best_model=True,
    )
    checkpoint_path = os.path.join(output_dir, "checkpoint-best.pt")
    assert os.path.exists(checkpoint_path)

    # Assert that the best model checkpoint file contains the expected data
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["model_config"] == model.config
    for key in checkpoint["model_state"]:
        assert torch.equal(checkpoint["model_state"][key], model.state_dict()[key])
    for key in checkpoint["optimizer_state"]:
        assert checkpoint["optimizer_state"][key] == optimizer.state_dict()[key]
    for key in checkpoint["scheduler_state"]:
        assert checkpoint["scheduler_state"][key] == scheduler.state_dict()[key]
    assert checkpoint["scaler_state"] is None
    assert checkpoint["trainer_state"] == trainer_state
