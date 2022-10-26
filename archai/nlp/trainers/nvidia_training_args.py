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

    log_dir: str = field(default="~/logdir", metadata={"help": ""})

    use_cuda: bool = field(default=True, metadata={"help": ""})
    
    multi_gpu: str = field(default=None, metadata={"help": ""})
    
    fp16: bool = field(default=False, metadata={"help": ""})
    
    log_all_ranks: bool = field(default=False, metadata={"help": ""})
    
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

    type: str = field(default='jitlamb', metadata={"help": ""})
    
    lr: float = field(default=0.01, metadata={"help": ""})
    
    weight_decay: float = field(default=0.0, metadata={"help": ""})
    
    momentum: float = field(default=0.0, metadata={"help": ""})
    
    clip: float = field(default=0.25, metadata={"help": ""})
    
    clip_nonemb: bool = field(default=False, metadata={"help": ""})
    
    sample_softmax: int = field(default=-1, metadata={"help": ""})

    type: str = field(default='cosine', metadata={"help": ""})
    
    type_qat: str = field(default='cosine', metadata={"help": ""})
    
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
    
    tgt_len: int = field('_copy: bool = field(/train/loader/tgt_len', metadata={"help": ""})
    
    mem_len: int = field('_copy: bool = field(/train/loader/mem_len', metadata={"help": ""})
    
    ext_len: int = field('_copy: bool = field(/train/loader/ext_len', metadata={"help": ""})

    name: str = field(default='olx_OutlookData20210917x2', metadata={"help": ""})
    
    max_batches: int = field(default=-1, metadata={"help": ""})
    
    cache_dir: str = field(default='cache_prefix', metadata={"help": ""})
    
    vocab: str = field(default="gpt2", metadata={"help": ""})
    
    vocab_size: int = field(default=10000, metadata={"help": ""})
    
    data_dir: str = field(default='textpred/olx_OutlookData20210917x2', metadata={"help": ""})


    def __post_init__(self) -> None:
        """
        """
        
        exp_utils.script_init()

        # Initializes distributed backend
        torch.cuda.set_device(self.local_rank)
        l2_promote()

        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        nv_distributed.init_distributed(self.use_cuda)

        self.data, self.work_dir, self.pretrained_path, self.cache_dir, self.dataroot = \
            exp_utils.get_create_dirs(self.data, self.dataset, self.experiment_name,
                                    self.work_dir, self.pretrained_path, self.cache_dir)

        with nv_distributed.sync_workers() as rank:
            if rank == 0:
                create_exp_dir(
                    self.work_dir,
                    scripts_to_save=[], #['train.py', 'mem_transformer.py'],
                    debug=self.debug
                )

        # Setup logging
        if self.log_all_ranks:
            log_file = f'train_log_rank_{nv_distributed.get_rank()}.log'
        else:
            log_file = self.txtlog_file
        dllog_file = self.dllog_file
        log_file = os.path.join(self.work_dir, log_file)
        dllog_file = os.path.join(self.work_dir, dllog_file)

        # if self.debug:
        #     log_file = os.devnull
        #     dllog_file = os.devnull

        exp_utils.setup_logging(log_all_ranks=self.log_all_ranks, filename=log_file)
        exp_utils.setup_dllogger(enabled=True, filename=dllog_file, disable_multiple=disable_multiple_dlogger)

        if self.config == 'toy':
            logging.warning('Running in toy mode which means wt2 dataset, only one step training, a lot of batch chunking for laptop GPU')

        if self.local_batch_size is not None: # default is None
            world_size = nv_distributed.get_world_size()
            self.batch_size = world_size * self.local_batch_size
            logging.info(f'--local_batch_size was set, adjusting global batch size'
                        f' to {self.batch_size} (local_batch_size * world_size)')

        logging.info(self)
        dllogger.log(step='PARAMETER', data=vars(self))

        logging.info(f'world size: {nv_distributed.get_world_size()}')

        if not self.debug and not self.no_env:
            log_env_info()

        #register_ignoring_timeout_handler()

        # Set the random seed manually for reproducibility.
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        logging.info('=' * 100)
        for k, v in self.__dict__.items():
            logging.info('    - {} : {}'.format(k, v))
        logging.info('=' * 100)

        if self.mem_len == 0: # default is 192
            self.eval_mem_len = 0
        else:
            self.eval_mem_len = self.mem_len + self.tgt_len - self.eval_tgt_len