# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
import torch
import numpy as np

from dataclasses import dataclass, field

from archai.nlp.datasets.nvidia_datasets import exp_utils
from archai.nlp.datasets.nvidia_datasets.distributed_utils import backend


@dataclass
class NvidiaTrainingArguments:
    """
    """

    experiment_name: str = field(metadata={"help": "Name of the experiment."})

    experiment_desc: str = field(default="", metadata={"help": "Description of the experiment."})

    log_dir: str = field(default="~/logdir", metadata={"help": ""})

    use_cuda: bool = field(default=True, metadata={"help": ""})
    
    multi_gpu: str = field(default=None, metadata={"help": ""})
    
    fp16: bool = field(default=False, metadata={"help": ""})

    local_rank: int = field(default=os.getenv('LOCAL_RANK', 0), metadata={"help": ""})
    
    log_all_ranks: bool = field(default=False, metadata={"help": ""})

    disable_multiple_dlogger: bool = field(default=False, metadata={"help": ""})
    
    txtlog_file: str = field(default='train_log.log', metadata={"help": ""})
    
    dllog_file: str = field(default='train_log.json', metadata={"help": ""})
    
    debug: bool = field(default=False, metadata={"help": ""})
    
    seed: int = field(default=42, metadata={"help": ""})
    
    toy: bool = field(default=False, metadata={"help": ""})
    
    no_train: bool = field(default=False, metadata={"help": ""})
    
    refresh_cache: bool = field(default=False, metadata={"help": ""})

    max_step: int = field(default=40000, metadata={"help": ""})
    
    batch_chunk: int = field(default=1, metadata={"help": ""})
    
    gpu0_bsz: int = field(default=-1, metadata={"help": ""})
    
    swap_mem: bool = field(default=False, metadata={"help": ""})
    
    mixed_qat: bool = field(default=False, metadata={"help": ""})
    
    qat: bool = field(default=False, metadata={"help": ""})
    
    pretrained_path: str = field(default="", metadata={"help": ""})

    batch_size: int = field(default=256, metadata={"help": ""})
    
    local_batch_size: int = field(default=None, metadata={"help": ""})
    
    roll: bool = field(default=False, metadata={"help": ""})
    
    varlen: bool = field(default=False, metadata={"help": ""})
    
    tgt_len: int = field(default=192, metadata={"help": ""})
    
    mem_len: int = field(default=0, metadata={"help": ""})
    
    ext_len: int = field(default=0, metadata={"help": ""})

    optimizer_type: str = field(default='jitlamb', metadata={"help": ""})
    
    lr: float = field(default=0.01, metadata={"help": ""})
    
    weight_decay: float = field(default=0.0, metadata={"help": ""})
    
    momentum: float = field(default=0.0, metadata={"help": ""})
    
    clip: float = field(default=0.25, metadata={"help": ""})
    
    clip_nonemb: bool = field(default=False, metadata={"help": ""})
    
    sample_softmax: int = field(default=-1, metadata={"help": ""})

    scheduler_type: str = field(default='cosine', metadata={"help": ""})
    
    qat_scheduler_type: str = field(default='cosine', metadata={"help": ""})
    
    max_step_scheduler: int = field(default=None, metadata={"help": ""})
    
    warmup_step: int = field(default=1000, metadata={"help": ""})
    
    patience: float = field(default=0, metadata={"help": ""})
    
    eta_min: float = field(default=0.001, metadata={"help": ""})
    
    lr_min: float = field(default=0.0, metadata={"help": ""})
    
    decay_rate: float = field(default=0.5, metadata={"help": ""})

    interval: int = field(default=5000, metadata={"help": ""})
    
    no_eval: bool = field(default=False, metadata={"help": ""})
    
    max_steps: int = field(default=-1, metadata={"help": ""})

    batch_size: int = field(default=256, metadata={"help": ""})
    
    local_batch_size: int = field(default=None, metadata={"help": ""})

    dataset: str = field(default='olx_OutlookData20210917x2', metadata={"help": ""})
    
    max_batches: int = field(default=-1, metadata={"help": ""})
    
    cache_dir: str = field(default='cache_prefix', metadata={"help": ""})
    
    vocab: str = field(default="gpt2", metadata={"help": ""})
    
    vocab_size: int = field(default=10000, metadata={"help": ""})
    
    data_dir: str = field(default='textpred/olx_OutlookData20210917x2', metadata={"help": ""})


    def __post_init__(self) -> None:
        """
        """
        
        exp_utils.script_init()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        if self.use_cuda:
            torch.cuda.set_device(self.local_rank)
            exp_utils.l2_promote()
            backend.init_distributed(self.use_cuda)

        self.data, self.log_dir, self.pretrained_path, self.cache_dir, self.dataroot = \
            exp_utils.get_create_dirs(self.data_dir, self.dataset, self.experiment_name,
                                    self.log_dir, self.pretrained_path, self.cache_dir)

        with backend.sync_workers() as rank:
            if rank == 0:
                exp_utils.create_exp_dir(
                    self.log_dir,
                    scripts_to_save=[], #['train.py', 'mem_transformer.py'],
                    debug=self.debug
                )

        if self.log_all_ranks:
            log_file = f'train_log_rank_{backend.get_rank()}.log'
        else:
            log_file = self.txtlog_file
        dllog_file = self.dllog_file
        log_file = os.path.join(self.log_dir, log_file)
        dllog_file = os.path.join(self.log_dir, dllog_file)

        exp_utils.setup_logging(log_all_ranks=self.log_all_ranks, filename=log_file)
        exp_utils.setup_dllogger(enabled=True, filename=dllog_file, disable_multiple=self.disable_multiple_dlogger)

        if self.local_batch_size is not None: # default is None
            world_size = backend.get_world_size()
            self.batch_size = world_size * self.local_batch_size

        #
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        #
        if self.mem_len == 0:
            self.eval_mem_len = 0
        else:
            self.eval_mem_len = self.mem_len + self.tgt_len - self.eval_tgt_len