# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer using NVIDIA-based pipeline.
"""

import argparse
import copy
import functools
import itertools
import logging
import math
import os
import pprint
import shutil
import sys
import time
from datetime import datetime
from packaging import version
from typing import Tuple

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from archai.common import ml_perf_utils, utils
from archai.nlp.datasets.nvidia_datasets import exp_utils
from archai.nlp.datasets.nvidia_datasets.distributed_utils import backend
from archai.nlp.datasets.nvidia_datasets.distributed_utils.data_parallel import BalancedDataParallel
from archai.nlp.datasets.nvidia_datasets.distributed_utils.data_utils import get_lm_corpus
from archai.nlp.datasets.nvidia_datasets.exp_utils import (AverageMeter, create_exp_dir, l2_promote,
                                           log_env_info)
from torch.nn.parallel import DistributedDataParallel

from archai.nlp.trainers.nvidia_training_args import NvidiaTrainingArguments


class NvidiaTrainer:
    """
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
    ) -> None:
        """"""

        assert isinstance(args, NvidiaTrainingArguments), "`args` should be an instance of `NvidiaTrainingArguments`."
        self.args = args
        
        self.dataset = get_lm_corpus(
            self.args.data,
            self.args.cache_dir,
            self.args.dataset,
            self.args.vocab,
            vocab_size=self.args.vocab_size,
            refresh_cache=self.args.refresh_cache
        )

    def _get_dataloader(self, split: str):
        """"""

        return self.dataset.get_iterator(
            split,
            self.args.batch_size,
            self.args.tgt_len,
            self.args.device,
            self.args.ext_len,
            mem_len=self.args.mem_len
        )

    def create_or_load_model(self):
        """"""

        model_config = {
            'n_token': ntokens,
            'n_layer': self.args.n_layer,
            'n_head': self.args.n_head,
            'd_model': self.args.d_self.model,
            'd_head': self.args.d_head,
            'd_inner': self.args.d_inner,
            'dropout': self.args.dropout,
            'dropatt': self.args.dropatt,
            'dtype': None,
            'tie_weight': self.args.tied,
            'd_embed': self.args.d_embed,
            'div_val': self.args.div_val,
            'tie_projs': tie_projs,
            'pre_lnorm': self.args.pre_lnorm,
            'tgt_len': self.args.tgt_len,
            'ext_len': self.args.ext_len,
            'mem_len': self.args.mem_len,
            'cutoffs': cutoffs,
            'adaptive': self.args.adaptive,
            'same_length': self.args.same_length,
            'attn_type': self.args.attn_type,
            'clamp_len': self.args.clamp_len,
            'sample_softmax': self.args.sample_softmax,

            'weight_init_type': self.args.init,
            'weight_init_range': self.args.init_range,
            'weight_init_std': self.args.init_std,
            'proj_init_std': self.args.proj_init_std,

            'primer_square': self.args.primer_square,
            'primer_conv': self.args.primer_conv,
            'use_cache': self.args.use_cache
            }

        if self.args.qat and self.args.mixed_qat:
            raise ValueError('QAT and Mixed QAT cannot be used at the same time.')

        if self.args.pretrained_path:
            self.model, model_config, _ = load_model_from_checkpoint(self.args.model_type, self.args.pretrained_path, on_cpu=False)
        else:
            model = load_model_from_config(self.args.model_type, model_config)

        if self.args.mixed_qat:
            model = MixedQATModel(model)

        n_params = model.get_params()
        n_all_param = n_params['total']
        n_nonemb_param = n_params['non_embedding']

        if self.args.qat:
            model = prepare_with_qat(self.model, onnx_compatible=True)

    def wrap_distributed_model(self):
        """"""
        if self.args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
            para_model = DistributedDataParallel(self.model,
                                                device_ids=[self.args.local_rank],
                                                output_device=self.args.local_rank,
                                                broadcast_buffers=False,
                                                find_unused_parameters=utils.is_debugging(),
                                                )
        elif self.args.multi_gpu == 'dp':
            if self.args.gpu0_bsz >= 0:
                para_model = BalancedDataParallel(self.args.gpu0_bsz // self.args.batch_chunk,
                                                self.model, dim=1).to(self.device)
            else:
                para_model = nn.DataParallel(self.model, dim=1).to(self.device)
        else:
            para_model = model

    def create_optimizer(self):
        """"""
        if self.args.optim.lower() == 'sgd':
            if self.args.sample_softmax > 0:
                dense_params, sparse_params = [], []
                for param in self.model.parameters():
                    if param.size() == self.model.word_emb.weight.size():
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                optimizer_sparse = optim.SGD(sparse_params, lr=self.args.lr * 2)
                optimizer = optim.SGD(dense_params, lr=self.args.lr, momentum=self.args.mom)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,
                                    momentum=self.args.mom)
                optimizer_sparse = None
        elif self.args.optim.lower() == 'adam':
            if self.args.sample_softmax > 0:
                dense_params, sparse_params = [], []
                for param in self.model.parameters():
                    if param.size() == self.model.word_emb.weight.size():
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                optimizer_sparse = optim.SparseAdam(sparse_params, lr=self.args.lr)
                optimizer = optim.Adam(dense_params, lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                optimizer_sparse = None
        elif self.args.optim.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr)
            optimizer_sparse = None
        elif self.args.optim.lower() == 'lamb':
            optimizer = lamb_optimizer.Lamb(self.model.parameters(), lr=self.args.lr,
                                weight_decay=self.args.weight_decay)
            optimizer_sparse = None
        elif self.args.optim.lower() == 'jitlamb':
            optimizer = lamb_optimizer.JITLamb(self.model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
            optimizer_sparse = None
        else:
            raise NotImplementedError(f'Optimizer {self.args.optim} is not implemented')

    def create_scaler(self):
        """"""

        scaler = None
        if self.args.fp16:
            scaler = torch.cuda.amp.GradScaler()
        return scaler

    def create_scheduler(self):
        """"""
        scheduler, scheduler_sparse = None, None
        scheduler_name = self.args.scheduler_qat if self.args.qat else self.args.scheduler

        # scheduler
        if scheduler_name == 'cosine':
            if self.args.max_step_scheduler:
                max_step = self.args.max_step_scheduler
            else:
                max_step = self.args.max_step

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max_step - self.args.warmup_step, eta_min=self.args.eta_min)
            if self.args.sample_softmax > 0 and optimizer_sparse is not None:
                scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_sparse, max_step - self.args.warmup_step,
                    eta_min=self.args.eta_min)
            else:
                scheduler_sparse = None
        elif scheduler_name == 'inv_sqrt':
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.args.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > self.args.warmup_step \
                        else step / (self.args.warmup_step ** 1.5)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            if self.args.sample_softmax > 0 and optimizer_sparse is not None:
                scheduler_sparse = optim.lr_scheduler.LambdaLR(
                    optimizer_sparse,
                    lr_lambda=lr_lambda
                    )
            else:
                scheduler_sparse = None
        elif scheduler_name == 'dev_perf':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.decay_rate, patience=self.args.patience,
                min_lr=self.args.lr_min,
                )
            if self.args.sample_softmax > 0 and optimizer_sparse is not None:
                scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_sparse, factor=self.args.decay_rate, patience=self.args.patience,
                    min_lr=self.args.lr_min,
                    )
            else:
                scheduler_sparse = None
        elif scheduler_name == 'cyclic_cosine':
            init_decay_epochs = int((self.args.max_step-self.args.warmup_step) / 2)
            restart_interval = int((self.args.max_step-self.args.warmup_step) / 4)

            scheduler = CyclicCosineDecayLR(optimizer, init_decay_epochs, self.args.eta_min, restart_interval, 
                                            warmup_epochs=self.args.warmup_step, warmup_start_lr=self.args.lr*0.01)
            if self.args.sample_softmax > 0 and optimizer_sparse is not None:
                scheduler_sparse = CyclicCosineDecayLR(optimizer_sparse, init_decay_epochs, self.args.eta_min, restart_interval, 
                                            warmup_epochs=self.args.warmup_step, warmup_start_lr=self.args.lr*0.01)
            else:
                scheduler_sparse = None
        elif scheduler_name == 'constant':
            pass

    def training_step(self):
        """"""
        self.model.train()

        train_loss = 0
        labels_tokens = 0
        log_step = 0
        log_start_time = time.time()

        mems = [None for _ in range(self.args.batch_chunk)]
        # Changes to make train_iter for lm1b to be properly caught
        if self.args.dataset != 'lm1b':
            if self.args.varlen:
                train_iter = train_itr.get_varlen_iter(start=last_iter)
            else:
                train_iter = train_itr.get_fixlen_iter(start=last_iter)
        else:
            train_iter = train_itr

        # Supports different autocast signatures and usage of bfloat16
        autocast = torch.cuda.amp.autocast(enabled=self.args.fp16)
        if version.parse(torch.__version__) >= version.parse('1.10'):
            fp16_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=fp16_type)

        for batch, (input_ids, labels, seq_len, _) in enumerate(train_iter, start=last_batch+1):
            log_step += 1
            labels_tokens += labels.numel()

            for param in self.model.parameters():
                param.grad = None

            # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
            input_ids_chunks = torch.chunk(input_ids, self.args.batch_chunk, 0)
            labels_chunks = torch.chunk(labels, self.args.batch_chunk, 0)

            for i in range(self.args.batch_chunk):
                # if this is last chunk and distribued mode then use delay_unscale=True for amp
                if i < self.args.batch_chunk - 1 and isinstance(para_self.model, DistributedDataParallel):
                    with para_model.no_sync():
                        train_loss_chunk = train_iteration(
                            para_self.model, i, mems, input_ids_chunks, labels_chunks, scaler,
                            optimizer, device, True, args, autocast
                        )
                else:
                    train_loss_chunk = train_iteration(
                        para_self.model, i, mems, input_ids_chunks, labels_chunks, scaler,
                        optimizer, device, False, args, autocast
                    )

                train_loss += train_loss_chunk

            if self.args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            if self.args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                if optimizer_sparse:
                    optimizer_sparse.step()

            # step-wise learning rate annealing
            train_step += 1
            if self.args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if train_step < self.args.warmup_step:
                    curr_lr = self.args.lr * train_step / self.args.warmup_step
                    optimizer.param_groups[0]['lr'] = curr_lr
                    if optimizer_sparse:
                        optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
                else:
                    if self.args.scheduler == 'cosine':
                        scheduler.step(train_step - self.args.warmup_step)
                        if scheduler_sparse:
                            scheduler_sparse.step(train_step - self.args.warmup_step)
            elif self.args.scheduler in ['inv_sqrt', 'cyclic_cosine']:
                scheduler.step(train_step)
                if scheduler_sparse:
                    scheduler_sparse.step(train_step)

            if train_step % self.args.log_interval == 0:
                cur_loss = train_loss / log_step
                cur_loss = nv_distributed.all_reduce_item(cur_loss, op='mean')
                train_loss = 0

                elapsed = time.time() - log_start_time
                avg_elapsed = elapsed / log_step
                avg_elapsed = nv_distributed.all_reduce_item(avg_elapsed, op='max')
                log_start_time = time.time()
                log_step = 0

                lr = optimizer.param_groups[0]['lr']
                throughput = labels_tokens / elapsed
                throughput = nv_distributed.all_reduce_item(throughput, op='sum')
                meters['train_throughput'].update(throughput)
                labels_tokens = 0

            do_periodic_eval = train_step % self.args.eval_interval == 0
            is_final_step = train_step == self.args.max_step
            interrupted = False #timeout_handler.interrupted

            if (do_periodic_eval or is_final_step or interrupted) and not self.args.no_eval:
                eval_start_time = time.time()
                node_metrix = evaluate(valid_itr, self.model, args, eval_nomem=False)
                val_metrix = EvalMetrics(valid_file_stats.word_count, *node_metrix)

                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                        '| loss {:5.2f} | word ppl {:5.2f}'.format(
                            train_step // self.args.eval_interval,
                            train_step,
                            (time.time() - eval_start_time),
                            val_metrix.avg_loss, val_metrix.word_ppl
                            )

                dllogger_data = {
                    'valid_elapsed': (time.time() - eval_start_time),
                    'valid_loss': val_metrix.avg_loss,
                    'valid_ppl': val_metrix.ppl,
                    'valid_word_ppl': val_metrix.word_ppl
                    }

                if self.args.dataset in ['enwik8', 'text8']:
                    log_str += ' | bpc {:9.5f}'.format(val_metrix.bpc)
                    dllogger_data['valid_bits_per_character'] = val_metrix.bpc
                else:
                    log_str += ' | ppl {:9.3f}'.format(val_metrix.ppl)
                    dllogger_data['valid_perplexity'] = val_metrix.ppl

                last_iter = train_itr.last_iter

                # Check if the validation loss is the best we've seen so far.
                is_best = False
                if not best_val_loss or val_metrix.avg_loss < best_val_loss:
                    best_val_loss = val_metrix.avg_loss
                    is_best = True

                model_to_save = model
                prefix = ''

                if self.args.qat:
                    # Convert the model to a regular FP32 model for saving
                    model_float = copy.deepcopy(self.model)
                    model_float = qat_to_float_modules(model_float)
                    model_to_save = model_float
                    prefix = 'qat_'

                save_checkpoint(args, model_to_save, model_config, optimizer, scheduler,
                                scaler, vocab, epoch, batch, last_iter,
                                train_step, best_val_loss, is_best,
                                self.args.work_dir, prefix=prefix)

                # dev-performance based learning rate annealing
                if self.args.scheduler == 'dev_perf':
                    scheduler.step(val_metrix.avg_loss)
                    if scheduler_sparse:
                        scheduler_sparse.step(val_metrix.avg_loss)

                # subtract eval time from timers for training
                log_start_time += time.time() - eval_start_time

            if interrupted:
                sys.exit(0)

            if is_final_step:
                break

    def train(self):
        """"""
        train_step = 0
        start_epoch = 1
        last_batch = 0
        last_iter = 0
        best_val_loss = None

        if self.args.restart:
            try:
                self.model, model_config, checkpoint = load_model_from_checkpoint(self.args.model_type, self.args.restart, on_cpu=False)
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                if self.args.fp16:
                    scaler.load_state_dict(checkpoint['amp_state'])
                train_step = checkpoint['train_step']
                start_epoch = checkpoint['epoch']
                last_batch = checkpoint['batch']
                last_iter = checkpoint['last_iter']
                best_val_loss = checkpoint['best_val_loss']

                if train_step >= self.args.max_step:
                    sys.exit(1)

                model.apply(functools.partial(update_dropout, args=args))
                model.apply(functools.partial(update_dropatt, args=args))
                
                para_self.model, model = distributed_model(args, self.model, self.device)
            except FileNotFoundError:
                pass

        meters = {}
        warmup = self.args.mem_len // self.args.tgt_len + 2
        meters['train_throughput'] = AverageMeter(warmup=warmup)
        ###########################################################################
        # Train
        ###########################################################################
        # Loop over epochs.
        # At any point you can hit Ctrl + C to break out of training early.
        start_time = time.time()
        try:
            for epoch in itertools.count(start=start_epoch):
                if self.args.roll: # enable random shifts in datasets
                    train_itr.roll(seed=self.args.seed + epoch)
                train_step, best_val_loss = train(
                    train_itr, valid_itr, self.model, para_self.model, model_config,
                    optimizer, optimizer_sparse, scheduler,
                    scheduler_sparse, scaler, vocab, epoch, last_batch,
                    last_iter, train_step, best_val_loss, meters,
                    device, args, valid_file_stats
                    )

                last_batch = 0
                last_iter = 0

                if train_step == self.args.max_step:
                    break

            if self.args.dynamic_quantization:
                dynamic_quantization_torch_from_model(model.cpu())

                save_checkpoint(args, self.model, model_config, optimizer, scheduler,
                                scaler, vocab, epoch, last_batch, last_iter,
                                train_step, best_val_loss, False,
                                self.args.work_dir, prefix='qnt-')

        except KeyboardInterrupt:
            pass

        elapsed = time.time() - start_time

    def evaluation_step(self):
        """"""
        self.model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        # default mem_len==192, eval_tgt_len==192, tgt_len==192
        if self.args.mem_len == 0:
            self.model.reset_length(tgt_len=self.args.eval_tgt_len,
                            ext_len=self.args.ext_len + self.args.tgt_len - self.args.eval_tgt_len,
                            mem_len=self.args.mem_len
                            )
        else:
            self.model.reset_length(tgt_len=self.args.eval_tgt_len,
                            ext_len=self.args.ext_len,
                            mem_len=self.args.mem_len + self.args.tgt_len - self.args.eval_tgt_len,
                            )

        # Evaluation
        total_len, total_loss, total_loss_nomem, steps, total_len_nowarmup, batches = 0, 0., 0., 0, 0, -1
        start_time = time.time()
        with torch.no_grad():
            mems = None
            for batches, (input_ids, labels, seq_len, warm) in enumerate(eval_iter):
                steps += 1
                if self.args.eval_max_steps > 0 and i >= self.args.eval_max_steps:
                    break

                # first with mem
                loss, _, mems, _ = self.model(input_ids, labels, mems)
                loss = loss.float().mean()
                numel = input_ids.numel()

                # now without mem
                loss_nomem = None
                if eval_nomem:
                    loss_nomem, _, _, _ = self.model(input_ids, labels, None)
                    loss_nomem = loss_nomem.float().mean()

                total_len_nowarmup += numel
                if warm:
                    # assert (mems is None) or mems.size(1) == model.mem_len
                    total_loss += numel * loss.item()
                    total_len += numel

                    if eval_nomem:
                        total_loss_nomem += numel * loss_nomem.item()

        elapsed = time.time() - start_time

        # Switch back to the training mode
        model.reset_length(tgt_len=self.args.tgt_len,
                        ext_len=self.args.ext_len,
                        mem_len=self.args.mem_len
                        )
        model.train()


    def evaluate(self):
        """"""
        n_params = self.model.get_params()
        summary = {
            'n_all_param': n_params['total'],
            'n_nonemb_param': n_params['non_embedding']
        }

        if not self.args.no_eval and os.path.exists(checkpoint_path):
            # Load the best saved model
            self.model, _, _ = load_model_from_checkpoint(self.args.model_type, checkpoint_path, on_cpu=False)

            # Run on test data
            test_start_time = time.time()
            node_metrix = evaluate(test_itr, self.model, args, eval_nomem=True)
            test_metrix = EvalMetrics(test_file_stats.word_count, *node_metrix)

            test_elapsed = time.time() - test_start_time

            summary.update({
                'test_word_count': test_metrix.eval_word_count,
                'test_total_elapsed': test_metrix.total_elapsed,
                'test_elapsed': test_elapsed,
                'test_total_loss': test_metrix.total_loss,
                'test_total_loss_nomem': test_metrix.total_loss_nomem,
                'test_avg_loss': test_metrix.avg_loss,
                'test_avg_loss_nomem': test_metrix.avg_loss_nomem,
                'test_steps': test_metrix.total_steps,
                'test_len': test_metrix.total_len,
                'total_len_nowarmup': test_metrix.total_len_nowarmup,
                'warmup_discount': test_metrix.warmup_discount,
                'test_word_ppl': test_metrix.word_ppl,
                'test_word_ppl_nomem': test_metrix.word_ppl_nomem
                })

            if self.args.dataset in ['enwik8', 'text8']:
                summary['test_bits_per_character'] = test_metrix.bpc
                summary['test_bits_per_character_nomem'] = test_metrix.bpc_nomem
            else:
                summary['test_ppl'] = test_metrix.ppl
                summary['test_ppl_nomem'] = test_metrix.ppl_nomem

        return summary

    def post_train_with_qat(self):
        """"""
        # Creates a dictionary of replacement configs
        replace_model_config = {
            'dropout': 0.0,
            'dropatt': 0.0
        }

        # Loads the model from the best pre-trained checkpoint
        self.model, model_config, _ = load_model_from_checkpoint(self.args.model_type, checkpoint_path, replace_model_config=replace_model_config, on_cpu=False)

        # Prepares the model with QAT (also allows for distributed training)
        model = prepare_with_qat(self.model, onnx_compatible=True)
        model = model.to(self.device)
        para_self.model, model = distributed_model()

        # QAT-based arguments
        self.args.restart = None
        self.args.qat = True
        self.args.max_step = 10000
        self.args.lr = self.args.lr / 100
        self.args.eta_min = self.args.eta_min / 100
        self.args.eval_interval = 1000
        self.args.warmup_step = 1000
        self.args.optim = 'adam'

        # re-create optimizer
        optimizer, optimizer_sparse = create_optimizer()

        # re-create scheduler
        scheduler, scheduler_sparse = create_scheduler()

        # Performs a QAT fine-tuning
        training_time, best_val_loss, meters = train_main()
