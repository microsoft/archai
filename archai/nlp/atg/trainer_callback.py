from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
import time
import sys
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import shutil
import subprocess

class TrainerCallback(ProgressBarBase):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, smoothing=0.01):
        super().__init__()
        self.enabled = True
        self.steps = 0
        self.prev_avg_loss = None
        self.smoothing = smoothing


    def enabled(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)

        # clean up the GPU cache used for the benchmark
        # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4
        if self.steps == 0:
            torch.cuda.empty_cache()

        current_loss = float(trainer.progress_bar_dict["loss"])
        self.steps += 1
        avg_loss = 0
        if current_loss == current_loss:  # don't add if current_loss is NaN
            avg_loss = self.average_loss(
                current_loss, self.prev_avg_loss, self.smoothing
            )
            self.prev_avg_loss = avg_loss

        desc = f"Step:{self.steps}, Loss: {current_loss:.3f}, Avg: {avg_loss:.3f}, Time:{time.time()}"
        print(desc)


    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss
