'''
compute exact match results

python archai/nlp/nvidia_transformer_xl/exact_match.py --dataset wt2 --warmup_step 0 --max_step 1000 --eval_interval 100 --n_layer 8 --n_head 8 --d_head 32 --d_embed 256 --d_inner 1024 --mem_len 512 --tgt_len 512 --d_model 512 --dropout 0.1 --config dgx1_1gpu_fp16 --experiment_name transxl_char_exp3_wikifull_select_v1corrected --config_file char_no_fp.yaml --eval_tgt_len 1024

'''

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
import warnings
from tqdm import tqdm
from collections import Counter

import dllogger
import numpy as np
import torch
import yaml
try:
    import pyprof
except ModuleNotFoundError:
    warnings.warn('PyProf is unavailable')

from nvidia_utils.exp_utils import memstat_utils

from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl import nvidia_utils
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.data_utils import tokenize_raw
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import AverageMeter
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import benchmark
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import create_exp_dir
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import l2_promote
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import log_env_info


def parse_args():
    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)

    config_args, _ = cfg_parser.parse_known_args()

    if config_args.config is not None and config_args.config_file is not None:
        with open(config_args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['eval']
    else:
        config = {}

    parser.add_argument('--work_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    parser.add_argument('--cache_dir', default=None, type=str,
                         help='Directory to store dataset cache, if None then use data dir as parent')

    parser.add_argument('--manual', type=str, default=None, nargs='+',
                        help='run model on raw input data')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8', 'wt2'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--affinity', type=str,
                        default='single_unique',
                        choices=['socket', 'single', 'single_unique',
                                 'socket_unique_interleaved',
                                 'socket_unique_continuous',
                                 'disabled'],
                        help='type of CPU affinity')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling with DLProf')
    parser.add_argument('--type', type=str, default='pytorch',
                        choices=['pytorch', 'torchscript'],
                        help='type of runtime to use')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=64,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=640,
                        help='length of the retained previous heads')
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='max positional embedding index')
    parser.add_argument('--cuda', action='store_true',
                        help='Run evaluation on a GPU using CUDA')
    parser.add_argument('--model', type=str, default='',
                        help='path to the checkpoint')
    parser.add_argument('--manual_config', type=json.loads, default=None,
                        help='Manually specify config for the model')
    parser.add_argument('--manual_vocab', type=str, default='word',
                        choices=['word', 'bpe', 'char'],
                        help='Manually specify type of vocabulary')
    parser.add_argument('--fp16', action='store_true',
                        help='Run training in fp16/mixed precision')
    parser.add_argument('--log_all_ranks', action='store_true',
                        help='Enable logging for all distributed ranks')
    parser.add_argument('--dllog_file', type=str, default='eval_log.json',
                        help='Name of the DLLogger output file')
    parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
    parser.add_argument('--no_env', action='store_true',
                        help='Do not print info on execution env')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Report interval')
    parser.add_argument('--target_perplexity', type=float, default=None,
                        help='target perplexity')
    parser.add_argument('--target_throughput', type=float, default=None,
                        help='target throughput')
    parser.add_argument('--save_data', action='store_true',
                        help='save latency and throughput data to a file')
    parser.add_argument('--repeat', type=int, default=1,
                        help='loop over the dataset REPEAT times')
    parser.add_argument('--max_size', type=int, default=None,
                        help='run inference on up to MAX_SIZE batches')
    parser.add_argument('--percentiles', nargs='+', default=[90, 95, 99],
                        help='percentiles for latency confidence intervals')
    parser.add_argument('--save_torchscript', default=None, type=str,
                        help='save torchscript model to a file')
    parser.add_argument('--load_torchscript', default=None, type=str,
                        help='load torchscript model from a file')
    parser.add_argument('--local_rank',  type=int,
                        default=os.getenv('LOCAL_RANK', 0),
                        help='Used for multi-process training.')
    parser.add_argument('--experiment_name', default='nv_xformer_xl', type=str,
                         help='Directory for the results')

    parser.add_argument('--prompt_context_percent', type=float, default=-1,
                        help='Report interval')
    parser.add_argument('--generation_method', type=str, default='wt103',
                        choices=['greedy', 'pure', 'beam', 'topk', 'topp'],
                        help='dataset name')
    parser.add_argument('--beam_size', type=int, default=40,
                        help='Beam size')
    parser.add_argument('--topp', type=float, default=0.92,
                        help='p value for top-p (or nucleus) sampling')
    parser.add_argument('--topk', type=int, default=1,
                        help='k for top-k sampling')
    parser.add_argument('--num_prompts', type=int, default=500,
                        help='# prompts')
    parser.add_argument('--num_chars_generate', type=int, default=-1,
                        help='number of chars to generate')
    parser.add_argument('--prefix_len', type=int, default=-1,
                        help='length of prefix')

    # char model extensions
    #parser.add_argument('--model_ext', type=str, default="bert_style_word_segment",
    #                    help='extensions to trans-xl model')
    parser.add_argument('--suggestion_length', type=int, default=3,
                        help='suggestion length in terms of words')
    parser.add_argument('--exposure_num_prompt_tokens', type=int, default=10,
                        help='number of prompt tokens for quantifying exposure bias')
    parser.add_argument('--exposure_num_generation_tokens', type=int, default=10,
                        help='number of generation tokens for quantifying exposure bias')
    parser.add_argument('--memstat', action='store_true', help='call memstat')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()
    print(args)

    if args.manual:
        args.batch_size = 1

    if args.same_length and args.tgt_len > args.mem_len:
        warnings.warn('--same_length is intended to be used with large '
                      'mem_len relative to tgt_len')

    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')
    return args


def load_checkpoint(path, cuda):
    dst = f'cuda:{torch.cuda.current_device()}' if cuda else torch.device('cpu')
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def format_log(loss, split, args):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


def evaluate(eval_iter, model, meters, log_interval, max_size=None, repeat=1, num_characters=1, num_tokens=1, vocab_type="char", model_ext=None):
    total_len, total_loss = 0, 0.
    eval_step = 0

    log_throughput = 0
    log_latency = 0
    log_loss = 0
    idx = 0

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        mems = None
        for _ in range(repeat):
            for idx, batch_items in enumerate(eval_iter):
                if model_ext == "bert_style_word_segment" or model_ext == "char_emb_from_word":
                    (data, target, seq_len, warm, word_segment) = batch_items
                else:
                    (data, target, seq_len, warm) = batch_items
                if max_size and idx >= max_size:
                    break
                eval_step += 1

                torch.cuda.synchronize()
                start_iter = time.time()
                if model_ext == "bert_style_word_segment" or model_ext == "char_emb_from_word":
                    loss, mems = model(data, target, mems, word_segment)
                else:
                    loss, mems = model(data, target, mems)
                torch.cuda.synchronize()
                elapsed = time.time() - start_iter

                loss = loss.float().mean()
                log_loss += loss.item()
                if warm:
                    total_loss += seq_len * loss.item()
                    total_len += seq_len

                meters['eval_latency'].update(elapsed)
                log_latency += elapsed

                target_tokens = target.numel()
                throughput = target_tokens / elapsed
                throughput = nvidia_utils.distributed.all_reduce_item(throughput, op='sum')
                meters['eval_throughput'].update(throughput)
                log_throughput += throughput

                if eval_step % log_interval == 0:
                    log_throughput /= log_interval
                    log_latency /= log_interval
                    log_loss /= log_interval
                    log_ppl = math.exp(log_loss)

                    log_str = '| step {:>8d} | batches {:>6d} / {:d} ' \
                        '| ms/batch {:5.2f} | tok/s {:7.0f} | loss {:5.2f} | ppl {:5.2f}'.format(
                            eval_step,
                            idx+1,
                            eval_iter.n_batch,
                            log_latency * 1000,
                            log_throughput,
                            log_loss,
                            log_ppl,
                            )
                    logging.info(log_str)

                    #dllogger_data = {
                    #    'eval_latency': log_latency * 1000,
                    #    'eval_throughput': log_throughput,
                    #    'eval_loss': log_loss,
                    #    'eval_perplexity': log_ppl,
                    #    }
                    #dllogger.log(step=tuple([eval_step]), data=dllogger_data)

                    log_throughput = 0
                    log_latency = 0
                    log_loss = 0

    nvidia_utils.distributed.barrier()
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    logging.info('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))

    avg_loss = total_loss / total_len
    avg_loss = nvidia_utils.distributed.all_reduce_item(avg_loss, op='mean')
    if vocab_type == "char":
        bpc = avg_loss / math.log(2)
        word_ppl = math.pow(2, bpc * (num_characters / num_tokens))
        print("Loss = %.2f BPC = %.2f Word-PPL = %.2f Num-Chars = %d Num-Tokens = %d"%(avg_loss, bpc, word_ppl, num_characters, num_tokens))
    elif vocab_type == "word":
        word_ppl = math.exp(avg_loss)
        print("Loss = %.2f Word-PPL = %.2f"%(avg_loss, word_ppl))
    return avg_loss

# reference: https://github.com/lopuhin/transformer-xl/blob/fb11489ca4c6000573d27d5eaca3a641057c0a6a/pytorch/inference.py#L99
def get_log_probs(all_xs, model, device, tgt_len, model_ext=None, space_char_idx=None, memstat=None):
    """ Return log probabilities for next tokens.
    Shape of returned tensor is len(tokens) x len(self.vocab),
    where the first element contains log probabilities for tokens
    after the first, and last element log probabilities for tokens
    after the last one.
    """
    all_log_probs = []
    with torch.no_grad():
        mems = None #tuple()
        input_len = tgt_len #model.tgt_len
        for idx in range(0, len(all_xs), input_len):
            xs = all_xs[idx: idx + input_len]
            xs = xs.to(device=device)
            batch_dim = 1  # batch size dimension is 1
            xs = xs.unsqueeze(batch_dim)
            if model_ext == "bert_style_word_segment" or model_ext == "char_emb_from_word":
                word_segment = torch.full_like(xs, 0)
                word_idx = 1
                for j in range(word_segment.size(0)):
                    if xs[j,0] == space_char_idx:
                        word_segment[j,0] = 0
                        if j!=0:
                            word_idx += 1
                    else:
                        word_segment[j,0] = word_idx
                log_probs, mems = model(xs, None, mems, word_segment)
            else:
                if memstat:
                    memstat_utils("model_call_start")
                log_probs, mems = model(xs, None, mems, None, memstat=memstat)
                if memstat:
                    memstat_utils("model_call_end")
            log_probs = log_probs.squeeze(batch_dim).data.cpu()
            all_log_probs.append(log_probs)
    return torch.cat(all_log_probs)

def get_prefix_overlap_len(string_a, string_b):
    num_match = 0
    for ai, bi in zip(string_a, string_b):
        if ai == bi:
            num_match += 1
        else:
            return float(num_match/len(string_b))
    return float(num_match/len(string_b)) 

def generate(encoded, model, device, vocab, tgt_len, generation_method, beam_size, topp, topk, prompt_context_percent, vocab_type, model_ext, suggestion_length=None):
    exact_matches = Counter()
    total = Counter()
    partial_matches = Counter()
    start_time = time.time()
    with torch.no_grad():
        #pbar = tqdm(total=len(encoded))
        idx = 0
        for prompt_tensor, prompt_tokens, target_tokens in encoded:
            generated_text = []
            #while len("".join(generated_text).split()) < 4 and len(generated_text) < 100:
            while len(vocab.convert_to_text(generated_text, vocab_type).split()) < 4 and len(generated_text) < 100:
                # get log probs
                log_probs = get_log_probs(prompt_tensor, model, device, tgt_len, model_ext=model_ext, space_char_idx=vocab.get_idx(" "))[-1]
                top_indices = torch.argsort(log_probs)[-topk:]
                top_probs = log_probs[top_indices].double().exp()
                sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
                next_token = vocab.idx2sym[sampled_idx]

                # add to prompt_tensor
                next_token_tensor = torch.IntTensor(1)
                next_token_tensor[0] = sampled_idx
                prompt_tensor = torch.cat((prompt_tensor, next_token_tensor))
                generated_text.append(sampled_idx)

            # compute match metrics
            # generated_tokens = "".join(generated_text).split()
            generated_tokens = vocab.convert_to_text(generated_text, vocab_type).split()
            for suggestion_len in range(0, min(suggestion_length, len(target_tokens))):
                exact_match = True
                for token_i in range(0, suggestion_len+1):
                    if len(generated_tokens) < token_i + 1 or generated_tokens[token_i].lower() != target_tokens[token_i].lower():
                        exact_match = False
                        break
                if exact_match:
                    exact_matches[suggestion_len+1] += 1
                    #if suggestion_len > 2:
                    #    sys.exit(0)
                partial_matches[suggestion_len+1] += get_prefix_overlap_len(" ".join(generated_tokens[0:suggestion_len+1]), " ".join(target_tokens[0:suggestion_len+1]))
                total[suggestion_len+1] += 1
            print("Index: %d\nPrompt: %s\nGenerated Text: %s\nTarget Text: %s\n\n"%(idx, " ".join(prompt_tokens), vocab.convert_to_text(generated_text, vocab_type), " ".join(target_tokens)))
            #pbar.update(1)
            idx += 1
            #if idx > 5:
            #    break
        #pbar.close()
    res = ""
    for suggestion_len in range(1, len(total)+1):
        res += "%d: %.2f (%d/%d),"%(suggestion_len, float(exact_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, exact_matches[suggestion_len], total[suggestion_len])
    print("context=%.2f %s"%(prompt_context_percent, res))
    res = ""
    for suggestion_len in range(1, len(total)+1):
        res += "%d: %.2f (%d/%d),"%(suggestion_len, float(partial_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, partial_matches[suggestion_len], total[suggestion_len])
    print("context=%.2f %s"%(prompt_context_percent, res))
    end_time = time.time()
    print("Time = "+str(end_time-start_time))

def inference_latency(encoded, model, device, vocab, tgt_len, generation_method, beam_size, topp, topk, prompt_context_percent, vocab_type, num_chars_generate, suggestion_length, memstat=False):
    exact_matches = Counter()
    total = Counter()
    partial_matches = Counter()
    start_time = time.time()
    if memstat:
        memstat_utils("code_begin")
    with torch.no_grad():
        pbar = tqdm(total=len(encoded))
        idx = 0
        for prompt_tensor, prompt_tokens, target_tokens in encoded:
            if memstat:
                memstat_utils("sample_start")
            generated_text = []
            #while len("".join(generated_text).split()) < 4 and len(generated_text) < 100:
            while len(vocab.convert_to_text(generated_text, vocab_type)) < num_chars_generate and len(generated_text) < num_chars_generate:
                if memstat:
                    memstat_utils("timestep_start")
                # get log probs
                if memstat:
                    memstat_utils("get_log_probs_start")
                log_probs = get_log_probs(prompt_tensor, model, device, tgt_len, memstat=memstat)[-1]
                if memstat:
                    memstat_utils("get_log_probs_end")
                top_indices = torch.argsort(log_probs)[-topk:]
                top_probs = log_probs[top_indices].double().exp()
                sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
                next_token = vocab.idx2sym[sampled_idx]

                # add to prompt_tensor
                next_token_tensor = torch.IntTensor(1)
                next_token_tensor[0] = sampled_idx
                prompt_tensor = torch.cat((prompt_tensor, next_token_tensor))
                generated_text.append(sampled_idx)
                if memstat:
                    memstat_utils("timestep_end")
            pbar.update(1)
            idx += 1
            if memstat:
                memstat_utils("sample_end")
            #if idx > 5:
            #    break
        pbar.close()
    end_time = time.time()
    if memstat:
        memstat_utils("code_end")
    if not memstat:
        print("Time = "+str(end_time-start_time)+" # prompts = "+str(len(encoded))+ " num_chars_generate = "+str(num_chars_generate))

def exposure_ebm(encoded, model, device, vocab, tgt_len, generation_method, beam_size, topp, topk, prompt_context_percent, vocab_type, num_chars_generate, prefix_len, exposure_num_prompt_tokens=None, exposure_num_generation_tokens=None):
    with torch.no_grad():
        pbar = tqdm(total=len(encoded))
        idx = 0
        dp_generated_text = []
        mp_generated_text = []
        gold_generated_text = []
        for prompt_tensor, prompt_tokens, target_tokens, model_prefix_prompt_tensor, num_pref_tokens in encoded:
            data_prefix_generated_text = []
            while len(data_prefix_generated_text) < 100 and len(vocab.convert_to_text(data_prefix_generated_text, vocab_type).split()) < exposure_num_generation_tokens+1:
                # get log probs
                log_probs = get_log_probs(prompt_tensor, model, device, tgt_len)[-1]
                top_indices = torch.argsort(log_probs)[-topk:]
                top_probs = log_probs[top_indices].double().exp()
                sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
                next_token = vocab.idx2sym[sampled_idx]

                # add to prompt_tensor
                next_token_tensor = torch.IntTensor(1)
                next_token_tensor[0] = sampled_idx
                prompt_tensor = torch.cat((prompt_tensor, next_token_tensor))
                data_prefix_generated_text.append(sampled_idx)
            data_prefix_generated_text = " ".join(vocab.convert_to_text(data_prefix_generated_text, vocab_type).split()[0:exposure_num_generation_tokens])

            model_prefix_prompt_tensor = prompt_tensor
            while len(model_prefix_prompt_tensor) < len(prompt_tensor) + num_pref_tokens:
                # get log probs
                log_probs = get_log_probs(model_prefix_prompt_tensor, model, device, tgt_len)[-1]
                top_indices = torch.argsort(log_probs)[-topk:]
                top_probs = log_probs[top_indices].double().exp()
                sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
                next_token = vocab.idx2sym[sampled_idx]

                # add to prompt_tensor
                next_token_tensor = torch.IntTensor(1)
                next_token_tensor[0] = sampled_idx
                model_prefix_prompt_tensor = torch.cat((model_prefix_prompt_tensor, next_token_tensor))

            model_prefix_generated_text = []
            prompt_tensor = model_prefix_prompt_tensor
            while len(model_prefix_generated_text) < 100  and len(vocab.convert_to_text(model_prefix_generated_text, vocab_type).split()) < exposure_num_generation_tokens+1:
                # get log probs
                log_probs = get_log_probs(prompt_tensor, model, device, tgt_len)[-1]
                top_indices = torch.argsort(log_probs)[-topk:]
                top_probs = log_probs[top_indices].double().exp()
                sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
                next_token = vocab.idx2sym[sampled_idx]

                # add to prompt_tensor
                next_token_tensor = torch.IntTensor(1)
                next_token_tensor[0] = sampled_idx
                prompt_tensor = torch.cat((prompt_tensor, next_token_tensor))
                model_prefix_generated_text.append(sampled_idx)
            model_prefix_generated_text = " ".join(vocab.convert_to_text(model_prefix_generated_text, vocab_type).split()[0:exposure_num_generation_tokens])
            
            dp_generated_text.append(data_prefix_generated_text.strip())
            mp_generated_text.append(model_prefix_generated_text.strip())
            gold_generated_text.append(" ".join(target_tokens).strip())

            print(idx)
            print(data_prefix_generated_text.strip())
            print(model_prefix_generated_text.strip())
            print(" ".join(target_tokens).strip())

            pbar.update(1)
            idx += 1
            #if idx > 5:
            #    break
        pbar.close()
        import sacrebleu
        print(sacrebleu.corpus_bleu(dp_generated_text, [gold_generated_text]), len(gold_generated_text))
        print(sacrebleu.corpus_bleu(mp_generated_text, [gold_generated_text]), len(gold_generated_text))

def main():
    args = parse_args()
    if args.memstat:
        memstat_utils("program_start")

    if args.type == 'pytorch':
        from mem_transformer_inference import MemTransformerLM

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    nvidia_utils.distributed.init_distributed(args.cuda)

    args.data, args.work_dir, args.cache_dir = \
        exp_utils.get_create_dirs(args.data, args.dataset, args.experiment_name,
                                  args.work_dir, args.cache_dir)

    with nvidia_utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir, debug=args.debug)

    # Setup logging
    if args.log_all_ranks:
        log_file = f'eval_log_rank_{nvidia_utils.distributed.get_rank()}.log'
    else:
        log_file = f'eval_log.log'

    dllog_file = args.dllog_file
    log_file = os.path.join(args.work_dir, log_file)
    dllog_file = os.path.join(args.work_dir, dllog_file)
    if args.debug:
        log_file = os.devnull
        dllog_file = os.devnull

    #archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils.setup_logging(log_all_ranks=args.log_all_ranks,
    #                              filename=log_file,
    #                              filemode='a',
    #                              )
    #archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils.setup_dllogger(enabled=True, filename=dllog_file)

    if args.profile:
        try:
            pyprof.init(enable_function_stack=True)
        except NameError:
            warnings.warn('Called pyprof.init() but pyprof is not available')

    logging.info(args)
    #dllogger.log(step='PARAMETER', data=vars(args))

    if not args.no_env:
        log_env_info()

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model:
        model_path = args.model
    elif args.work_dir:
        model_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    else:
        raise RuntimeError('Specify path to checkpoint using --model or --work_dir')

    if True: #not args.manual_config:
        checkpoint = load_checkpoint(model_path, args.cuda)
        vocab_type = checkpoint['args'].vocab
    else:
        checkpoint = None
        vocab_type = args.manual_vocab

    if True: #args.manual:
        vocab = checkpoint['vocab']

        if hasattr(vocab, 'sym2idx') and not hasattr(vocab, 'unk_idx'):
            vocab.unk_idx = vocab.sym2idx['<unk>']
        
        '''
        text = "hello this is great hello this is great hello this is great hello this is great hello this is great" # " ".join(args.manual)
        tokenized = tokenize_raw(text)
        symbols = vocab.tokenize(tokenized, add_eos=False)
        tensor = vocab.convert_to_tensor(symbols)
        num_characters = len(text)
        num_tokens = len(text.split())
        iter = data_utils.LMOrderedIterator(tensor, bsz=args.batch_size,
                                            bptt=args.tgt_len, device=device,
                                            ext_len=args.ext_len, warmup=False)
        '''

        # process test file
        if args.prompt_context_percent <= 0.0:
            encoded = []
            if 'model_ext' in checkpoint['model_config'] and (checkpoint['model_config']['model_ext'] == "bert_style_word_segment" or checkpoint['model_config']['model_ext'] == "char_emb_from_word"):
                word_segment = []
            num_characters, num_tokens = 0, 0
            with open(args.data + "/wiki.test.tokens", 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    num_characters += len(line.strip())
                    num_tokens += len(line.strip().split())
                    symbols = vocab.tokenize(line, add_eos=False,
                                            add_double_eos=False)
                    encoded.append(vocab.convert_to_tensor(symbols))
                    if 'model_ext' in checkpoint['model_config'] and (checkpoint['model_config']['model_ext'] == "bert_style_word_segment"  or checkpoint['model_config']['model_ext'] == "char_emb_from_word"):
                        word_segment.append(vocab.convert_to_bert_style_segment(symbols))
            encoded = torch.cat(encoded)
            if 'model_ext' in checkpoint['model_config'] and (checkpoint['model_config']['model_ext'] == "bert_style_word_segment"  or checkpoint['model_config']['model_ext'] == "char_emb_from_word"):
                word_segment = torch.cat(word_segment)
                iter = data_utils.LMOrderedIteratorExt((encoded, word_segment), bsz=args.batch_size,
                                                bptt=args.tgt_len, device=device,
                                                ext_len=args.ext_len, warmup=False, model_ext=checkpoint['model_config']['model_ext'])
            else:
                iter = data_utils.LMOrderedIterator(encoded, bsz=args.batch_size,
                                                bptt=args.tgt_len, device=device,
                                                ext_len=args.ext_len, warmup=False)
        else:
            print('preparing prompts from %s/wiki.test.tokens'%(args.data))
            encoded = []
            #'''
            if args.prefix_len != -1:
                prompt_cache = {}
                with open(args.data + "/wiki.test.tokens", 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if len(line) == 0 or len(line.split()) <= 1:
                            continue
                        tokens = line.split()
                        if tokens[0] == "=":
                            continue
                        num_ptokens = args.exposure_num_prompt_tokens
                        prompt_tokens = tokens[0:num_ptokens]
                        if len(prompt_tokens) != num_ptokens:
                            continue
                        if " ".join(prompt_tokens) in prompt_cache:
                            continue
                        prompt_cache[" ".join(prompt_tokens)] = True
                        prefix_tokens = tokens[num_ptokens:num_ptokens+args.prefix_len]
                        if len(prefix_tokens) != args.prefix_len:
                            continue
                        target_tokens = tokens[num_ptokens+args.prefix_len:]
                        if len(target_tokens) == 0:
                            continue
                        target_tokens = tokens[num_ptokens:]
                        target_tokens = target_tokens[0:args.exposure_num_generation_tokens]
                        prompt_symbols = vocab.tokenize(" ".join(prompt_tokens + prefix_tokens) + " ", add_eos=False, add_double_eos=False)
                        prompt_tensor = vocab.convert_to_tensor(prompt_symbols)
                        model_prefix_prompt_tensor = vocab.convert_to_tensor( vocab.tokenize(" ".join(prompt_tokens) + " ", add_eos=False, add_double_eos=False))
                        encoded.append((prompt_tensor, prompt_tokens, target_tokens, model_prefix_prompt_tensor,  prompt_tensor.size(0) - model_prefix_prompt_tensor.size(0)))
                        if len(encoded) == args.num_prompts:
                            break
            else:
                prompt_cache = {}
                #with open(args.data + "/wiki.test.tokens", 'r', encoding='utf-8') as f:
                with open("/home/t-gjawahar/object_dir/WordData20210110/gan/tokenized_test.txt", 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if len(line) == 0 or len(line.split()) <= 1:
                            continue
                        tokens = line.split()
                        if tokens[0] == "=":
                            continue
                        num_ptokens = int(args.prompt_context_percent * len(tokens))
                        prompt_tokens = tokens[0:num_ptokens]
                        if " ".join(prompt_tokens) in prompt_cache:
                            continue
                        prompt_cache[" ".join(prompt_tokens)] = True
                        target_tokens = tokens[num_ptokens:]
                        target_tokens = target_tokens[0:args.suggestion_length]
                        if len(prompt_tokens) == 0 or len(target_tokens) == 0:
                            continue
                        prompt_symbols = vocab.tokenize(" ".join(prompt_tokens) + " ", add_eos=False,
                                                add_double_eos=False)
                        encoded.append((vocab.convert_to_tensor(prompt_symbols), prompt_tokens, target_tokens))
                        if len(encoded) == args.num_prompts:
                            break
            '''
            prompts = []
            # world knowledge
            prompts.append(("Paris is the capital of", "France"))
            prompts.append(("Olympia is the capital of", "Washington"))
            prompts.append(("Jessica Adele Hardy specializes in", "breaststroke and freestyle events")) 
            prompts.append(("The word ' jaguar ' comes to English from one of the", "Tupi – Guarani languages , presumably the Amazonian trade language Tupinambá , via Portuguese jaguar")) # present in training data
            prompts.append(("Pokiri ( English : Rogue ) is a 2006 Indian Telugu @-@ language action film , written and directed by", "Puri Jagannadh")) # present in training data
            # open-ended prompts
            prompts.append(("India is", ""))
            prompts.append(("Microsoft is", " "))
            # more context and open-ended prompt
            prompts.append(("Pokiri ( English : Rogue ) is a 2006 Indian Telugu @-@ language action film , written and directed by Puri Jagannadh .", "")) # present in training data
            prompts.append(("Gilchrist 's autobiography True Colours , published in 2008 , was the subject of much controversy . Gilchrist questioned the integrity of leading Indian batsman Sachin Tendulkar in relation to the evidence he presented in the Monkeygate dispute , which was about allegations of racism against Harbhajan Singh .", "")) # present in training data
            prompts.append(("Bradman 's Test batting average of 99 @.@ 94 has become one of cricket 's most famous , iconic statistics . No other player who has played more than 20 Test match innings has finished with a Test average of more than 61 . Bradman scored centuries at a rate better than one every three innings — in 80 Test innings", "")) # present in training data
            #prompts = [prompts[0]]

            for prompt_tokens, target_tokens in prompts:
                prompt_tokens = prompt_tokens.split()
                target_tokens = target_tokens.split()
                prompt_symbols = None
                if vocab_type == "word":
                    prompt_symbols = vocab.tokenize(" ".join(prompt_tokens), add_eos=False,
                                            add_double_eos=False)
                elif vocab_type == "char":
                    prompt_symbols =  ['<S>'] + vocab.tokenize(" " + " ".join(prompt_tokens), add_eos=False,
                                            add_double_eos=False)
                encoded.append((vocab.convert_to_tensor(prompt_symbols), prompt_tokens, target_tokens))
            #'''
    else:
        # Load dataset
        corpus = get_lm_corpus(args.data, args.cache_dir, args.dataset, vocab_type)

        if args.split == 'valid' or args.split == 'test':
            iter = corpus.get_iterator(args.split, args.batch_size, args.tgt_len,
                                       device=device, mem_len=args.mem_len,
                                       ext_len=args.ext_len)
        else:
            raise RuntimeError('Unknown split')

    if args.fp16:
        dtype = torch.float16
        math_str = 'fp16'
    else:
        dtype = torch.float32
        math_str = 'fp32'

    if args.load_torchscript:
        model = torch.jit.load(args.load_torchscript)
    elif not args.manual_config:
        checkpoint['model_config']['tgt_len'] = args.tgt_len
        checkpoint['model_config']['ext_len'] = args.ext_len
        checkpoint['model_config']['mem_len'] = args.mem_len
        checkpoint['model_config']['clamp_len'] = args.clamp_len
        checkpoint['model_config']['same_length'] = args.same_length
        checkpoint['model_config']['dtype'] = dtype

        model = MemTransformerLM(**checkpoint['model_config'])
        if args.type == 'pytorch':
            model.load_state_dict(checkpoint['model_state'])
        elif args.type == 'torchscript':
            model.load_state_dict(checkpoint['model_state'], strict=False)
    elif args.manual_config:
        args.manual_config['tgt_len'] = args.tgt_len
        args.manual_config['ext_len'] = args.ext_len
        args.manual_config['mem_len'] = args.mem_len
        args.manual_config['clamp_len'] = args.clamp_len
        args.manual_config['same_length'] = args.same_length
        args.manual_config['dtype'] = dtype

        model = MemTransformerLM(**args.manual_config)

    model = model.eval()
    model = model.to(device)
    model = model.to(dtype)

    logging.info(f'Evaluating with: math {math_str} type {args.type} '
                 f'bsz {args.batch_size} tgt_len {args.tgt_len} '
                 f'ext_len {args.ext_len} mem_len {args.mem_len} '
                 f'clamp_len {args.clamp_len}')

    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['eval_throughput'] = AverageMeter(warmup=warmup, keep=args.save_data)
    meters['eval_latency'] = AverageMeter(warmup=warmup, keep=args.save_data)

    with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
        if args.prompt_context_percent <= 0.0:
            loss = evaluate(iter, model, meters, args.log_interval, args.max_size,
                        args.repeat, num_characters=num_characters, num_tokens=num_tokens, vocab_type=vocab_type, model_ext=checkpoint['model_config']['model_ext'] if 'model_ext' in checkpoint['model_config'] else None)
        elif args.num_chars_generate != -1:
            inference_latency(encoded, model, device, vocab, args.tgt_len, args.generation_method, args.beam_size, args.topp, args.topk, args.prompt_context_percent, vocab_type=vocab_type, num_chars_generate=args.num_chars_generate, suggestion_length=args.suggestion_length, memstat=args.memstat)
        elif args.prefix_len != -1:
            exposure_ebm(encoded, model, device, vocab, args.tgt_len, args.generation_method, args.beam_size, args.topp, args.topk, args.prompt_context_percent, vocab_type=vocab_type, num_chars_generate=args.num_chars_generate, prefix_len=args.prefix_len, exposure_num_prompt_tokens=args.exposure_num_prompt_tokens, exposure_num_generation_tokens=args.exposure_num_generation_tokens)
        else:
            generate(encoded, model, device, vocab, args.tgt_len, args.generation_method, args.beam_size, args.topp, args.topk, args.prompt_context_percent, vocab_type=vocab_type, model_ext=checkpoint['model_config']['model_ext'] if 'model_ext' in checkpoint['model_config'] else None, suggestion_length=args.suggestion_length)

    #perplexity = math.exp(loss)
    #log_str = format_log(loss, args.split, args)
    if args.memstat:
        memstat_utils("program_end")

if __name__ == "__main__":
    # Disable profiling executor
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except AttributeError:
        pass

    main()