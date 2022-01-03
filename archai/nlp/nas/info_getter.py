# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Information getters for search-related tasks.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

plt.rcParams.update({'font.size': 18})

model_config_keys = ['n_token', 'n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'dropout', 'dropatt',
                     'd_embed', 'div_val', 'pre_lnorm', 'tgt_len', 'ext_len', 'mem_len',
                     'same_length', 'attn_type', 'clamp_len', 'sample_softmax',
                     'primer_conv', 'primer_square', 'use_cache']


def get_label(baseline_config, new_config):
    label = []

    for k in ['n_layer', 'd_model', 'd_inner']:
        if np.any(baseline_config[k] != new_config[k]):
            label.append(k)

    if 'n_layer' in label:
        label.remove('d_inner')

    return '_'.join(label)


def get_metrics(topk, sorted_ground_truth, sorted_target, val_ppl_list_gt, val_ppl_list_target):
    idx = int(topk/100.*len(sorted_ground_truth))

    sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
    sorted_target_binned = sorted_target[:idx].astype(np.int32)

    correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
    total = len(sorted_target_binned)
    common_ratio = correct*1./total

    print('Correctly ranked top %d %% (%d) with %.2f accuracy' %(topk, total, correct*1./total))

    topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_ground_truth_binned]
    topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_ground_truth_binned]
    spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)

    print('Spearman Correlation on top %d %% (%d): %.3f' % (topk, len(topk_val_ppl_list_gt), spr_rank))

    return common_ratio, spr_rank


def extract_date_time(log_str):
    idx_end_time = list(re.search(' - INFO - ', log_str).span())[0]
    idx_start_time = idx_end_time-8

    time_str = log_str[idx_start_time:idx_end_time]
    date_str = log_str[:idx_start_time-1]

    y, mo, d = date_str.split('-')
    h, m, s = time_str.split(':')

    return int(y), int(mo), int(d), int(h), int(m), int(s)


def extract_step(log_str):
    step = log_str.split('|')[1].split('step')[-1]

    return step


def get_info_from_logs(log_file, stage_1=True):
    out_dict = {}

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ppl_thresholds = None

    for idx, l in enumerate(lines):
        if 'ppl_threshold' in l:
            if ppl_thresholds is None:
                try:
                    str = re.search('\[(.+?)\]', l).group(1)
                    ppl_thresholds = [int(float(thr)) for thr in str.split(',')]
                except:
                    str = re.search('ppl_threshold=([0-9]+),', l).group(1)
                    ppl_thresholds = int(float(str))

        elif '#params' in l:
            n_params = re.search('#params = ([0-9]+)', l).group(1)
            out_dict['n_params'] = n_params

        elif '#non emb params' in l:
            start_y, start_mo, start_d, start_h, start_m, start_s = extract_date_time(lines[idx+1])

        elif 'Saving FEAR checkpoint' in l:
            end_y, end_mo, end_d, end_h, end_m, end_s = extract_date_time(l)

            idx_ppl = re.search('ppl', l).span()[-1]
            ppl = float(l.replace('\n', '')[idx_ppl:])
            ppl_key = min(ppl_thresholds, key=lambda x: abs(x-int(ppl)))

            out_dict[ppl_key] = {}
            out_dict[ppl_key]['ppl'] = ppl
            out_dict[ppl_key]['time'] = (end_y-start_y)*365*24*3600 + (end_mo-start_mo)*30*24*3600 + (end_d-start_d)*24*3600 + (end_h-start_h)*3600 + (end_m-start_m)*60 + (end_s-start_s)
            assert out_dict[ppl_key]['time'] > 0, print(end_y, end_mo, end_d, end_h, end_m, end_s, start_y, start_mo, start_d, start_h, start_m, start_s)

            step = int(extract_step(lines[idx-2]))

            out_dict[ppl_key]['step'] = step

        elif 'Training time:' in l and stage_1:
            t_minutes = re.search('Training time: ([+-]?([0-9]*[.])?[0-9]+) minutes', l).group(1)
            out_dict['train_time'] = float(t_minutes) * 60

    return out_dict


def get_info_from_json(json_file, step=[], type=None):
    out_dict = {}
    key = type+'_perplexity' if type is not None else None

    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[::-1]

        try:
            job_desc = re.search('DLLL \{(.+?)\}', lines[-1])
        except:
            return None

        job_desc = '{'+job_desc.group(1)+'}}'
        work_dir = json.loads(job_desc)['data']['work_dir']

        try:
            idx_start = re.search('amlt-results', work_dir).span()[-1] + 1
            amlt_job = work_dir[idx_start:].split('/')[0]
        except:
            amlt_job = None

        for line in lines:
            str = re.search('DLLL \{(.+?)\}', line)
            str = '{'+str.group(1)+'}}'

            final_train_log = json.loads(str)

            if len(step) > 0:
                if final_train_log['step'] == []:
                    out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
                    for k in final_train_log['data'].keys():
                        if 'perplexity' in k:
                            out_dict[k] = final_train_log['data'][k]

                elif final_train_log['step'][0] in step:
                    if final_train_log['step'][0] not in out_dict.keys():
                        out_dict[final_train_log['step'][0]] = {}
                    for k in final_train_log['data'].keys():
                        if 'perplexity' in k:
                            out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]

            else:
                try:
                    out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60

                    if key is None:
                        for k in final_train_log['data'].keys():
                            if 'perplexity' in k:
                                out_dict[k] = final_train_log['data'][k]

                    elif key in final_train_log['data'].keys():
                        out_dict[key] = final_train_log['data'][key]

                    out_dict['amlt_job'] = amlt_job
                    break

                except:
                    return None

    return out_dict


def get_config_name(job):
    if 'similar_params_sweep' in job or 'simp' in job or 'evolution' in job or 'lm1b' in job or 'corei' in job:
        idx = re.search('(config_[0-9]+)', job).span()[0]
        job = job[idx:]
        config_name = job.split('/')[0]

        return config_name + '_' + job.split('/')[1]

    elif 'baseline' in job:
        dir_name = os.path.basename(os.path.dirname(job))
        
        return re.search('(config_[0-9]+_[0-9]+)', dir_name).group(1)

    elif 'evo_search' in job or 'midevolution' in job:
        return job

    else:
        return re.search('(config_[0-9]+)', job).group(1)


def get_results(exp_name, path_to_dir, filetypes='.json', verbose=True):
    if not isinstance(filetypes, list):
        filetypes = [filetypes]

    results = {}

    for j in os.listdir(path_to_dir):
        j_path = os.path.join(path_to_dir, j)

        if os.path.isdir(j_path):
            results.update(get_results(exp_name, j_path, filetypes, verbose))

        else:
            for ft in filetypes:
                if ft in j:
                    if '.log' in ft:
                        logs = get_info_from_logs(j_path, stage_1='stage_1' in exp_name)
                    elif '.json' in ft:
                        logs = get_info_from_json(j_path)
                    elif '.yaml' in ft:
                        with open(os.path.join(j_path), 'r') as f:
                            config = yaml.load(f)

                        if config is None:
                            json_file = os.path.join(path_to_dir, 'train_log.json')

                            with open(json_file, 'r', encoding='utf-8') as f:
                                lines = f.readlines()

                                try:
                                    job_desc = re.search('DLLL \{(.+?)\}', lines[0])
                                except:
                                    return None

                                job_desc = '{'+job_desc.group(1)+'}}'
                                config = json.loads(job_desc)['data']
                                config['n_token'] = 267735

                        model_config = {k: config[k] for k in model_config_keys}

                        cutoffs, tie_projs = [], [False]

                        if config['adaptive']:
                            assert config['dataset'] in ['wt103', 'wt2', 'lm1b']

                            if config['dataset'] in ['wt103', 'wt2']:
                                cutoffs = [19997, 39997, 199997]
                                tie_projs += [True] * len(cutoffs)
                            elif config['dataset'] == 'lm1b':
                                cutoffs = [59997, 99997, 639997]
                                tie_projs += [False] * len(cutoffs)

                        model_config['cutoffs'] = cutoffs
                        model_config['tie_projs'] = tie_projs
                        model_config['tie_weight'] = config['tied']
                        model_config['dtype'] = None
                        logs = model_config

                    else:
                        assert False, 'unsupported file type {}'.format(ft)

                    if logs:
                        config_name = get_config_name(j_path)

                        if verbose:
                            print(config_name, logs)
                            
                        if config_name in results.keys():
                            results[config_name].update(logs)
                        else:
                            results[config_name] = logs

    return results
