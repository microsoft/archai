# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Parses incoming values, configurations and results.
"""

# import json
# import os
# import pickle
# import re
# import time
# from typing import Dict, List, Optional, Tuple
from typing import Any

# import matplotlib.pyplot as plt
# import numpy as np
# import yaml

# Keys that can be parsed from a model's configuration
# KEYS_MODEL_CONFIG = ['n_token', 'n_layer', 'n_head', 'd_model', 'd_head', 'd_inner',
#                      'dropout', 'dropatt', 'd_embed', 'div_val', 'pre_lnorm',
#                      'tgt_len', 'ext_len', 'mem_len', 'same_length', 'attn_type',
#                      'clamp_len', 'sample_softmax', 'primer_conv', 'primer_squared',
#                      'use_cache']


# def parse_config_name_from_job(job_str: str) -> str:
#     """Parses configuration name from a job string.

#     Args:
#         job_str: Log string to be parsed.

#     Returns:
#         (str): Parsed configuration name from the job string.

#     """

#     if 'similar_params_sweep' in job_str or 'simp' in job_str or 'evolution' in job_str or 'lm1b' in job_str or 'corei' in job_str:
#         idx = re.search('(config_[0-9]+)', job_str).span()[0]
#         job_str = job_str[idx:]
#         config_name = job_str.split('/')[0]

#         return config_name + '_' + job_str.split('/')[1]

#     elif 'baseline' in job_str:
#         dir_name = os.path.basename(os.path.dirname(job_str))
        
#         return re.search('(config_[0-9]+_[0-9]+)', dir_name).group(1)

#     elif 'evo_search' in job_str or 'midevolution' in job_str:
#         return job_str

#     else:
#         return re.search('(config_[0-9]+)', job_str).group(1)


# def parse_time_from_log(log_str: str) -> Tuple[int, ...]:
#     """Parses time (year, month, date, hour, minutes, seconds) from a log string.

#     Args:
#         log_str: Log string to be parsed.

#     Returns:
#         (Tuple[int, ...]): Parsed time from the log string.

#     """

#     idx_end_time = list(re.search(' - INFO - ', log_str).span())[0]
#     idx_start_time = idx_end_time - 8

#     time_str = log_str[idx_start_time:idx_end_time]
#     date_str = log_str[:idx_start_time - 1]

#     year, month, day = date_str.split('-')
#     hour, minute, second = time_str.split(':')

#     return int(year), int(month), int(day), int(hour), int(minute), int(second)


# def parse_step_from_log(log_str: str) -> int:
#     """Parses step value from a log string.

#     Args:
#         log_str: Log string to be parsed.

#     Returns:
#         (int): Parsed step from the log string.

#     """

#     step = log_str.split('|')[1].split('step')[-1]

#     return int(step)


# def parse_info_from_json_file(json_file: str,
#                               step: Optional[List[int]]=[],
#                               type: Optional[str] = None) -> Dict[str, Any]:
#     """Parses information from a JSON file.

#     Args:
#         json_file: Path to the JSON file.
#         step: Steps that should be parsed.
#         type: Type of perplexity value that should be parsed.

#     Returns:
#         (Dict[str, Any]): Parsed log file encoded into a dictionary.

#     """

#     out_dict = {}
#     key = type + '_perplexity' if type is not None else None

#     with open(json_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()[::-1]

#         try:
#             job_desc = re.search('DLLL \{(.+?)\}', lines[-1])
#         except:
#             return None

#         job_desc = '{'+job_desc.group(1)+'}}'
#         work_dir = json.loads(job_desc)['data']['work_dir']

#         try:
#             idx_start = re.search('amlt-results', work_dir).span()[-1] + 1
#             amlt_job = work_dir[idx_start:].split('/')[0]
#         except:
#             amlt_job = None

#         for line in lines:
#             str = re.search('DLLL \{(.+?)\}', line)
#             str = '{'+str.group(1)+'}}'

#             final_train_log = json.loads(str)

#             if len(step) > 0:
#                 if final_train_log['step'] == []:
#                     out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
#                     for k in final_train_log['data'].keys():
#                         if 'perplexity' in k:
#                             out_dict[k] = final_train_log['data'][k]

#                 elif final_train_log['step'][0] in step:
#                     if final_train_log['step'][0] not in out_dict.keys():
#                         out_dict[final_train_log['step'][0]] = {}
#                     for k in final_train_log['data'].keys():
#                         if 'perplexity' in k:
#                             out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]

#             else:
#                 try:
#                     out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60

#                     if key is None:
#                         for k in final_train_log['data'].keys():
#                             if 'perplexity' in k:
#                                 out_dict[k] = final_train_log['data'][k]

#                     elif key in final_train_log['data'].keys():
#                         out_dict[key] = final_train_log['data'][key]

#                     out_dict['amlt_job'] = amlt_job
#                     break

#                 except:
#                     return None

#     return out_dict


# def parse_info_from_log_file(log_file: str, stage_1: Optional[bool] = True) -> Dict[str, Any]:
#     """Parses information from a log file.

#     Args:
#         log_file: Path to the log file.
#         stage_1: Whether should parse information from FEAR stage 1 or not.

#     Returns:
#         (Dict[str, Any]): Parsed information encoded into a dictionary.

#     """

#     out_dict = {}

#     with open(log_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     ppl_thresholds = None

#     for idx, l in enumerate(lines):
#         if 'ppl_threshold' in l:
#             if ppl_thresholds is None:
#                 try:
#                     str = re.search('\[(.+?)\]', l).group(1)
#                     ppl_thresholds = [int(float(thr)) for thr in str.split(',')]
#                 except:
#                     str = re.search('ppl_threshold=([0-9]+),', l).group(1)
#                     ppl_thresholds = int(float(str))

#         elif '#params' in l:
#             n_params = re.search('#params = ([0-9]+)', l).group(1)
#             out_dict['n_params'] = n_params

#         elif '#non emb params' in l:
#             start_y, start_mo, start_d, start_h, start_m, start_s = parse_time_from_log(lines[idx+1])

#         elif 'Saving FEAR checkpoint' in l:
#             end_y, end_mo, end_d, end_h, end_m, end_s = parse_time_from_log(l)

#             idx_ppl = re.search('ppl', l).span()[-1]
#             ppl = float(l.replace('\n', '')[idx_ppl:])
#             ppl_key = min(ppl_thresholds, key=lambda x: abs(x-int(ppl)))

#             out_dict[ppl_key] = {}
#             out_dict[ppl_key]['ppl'] = ppl
#             out_dict[ppl_key]['time'] = (end_y-start_y)*365*24*3600 + (end_mo-start_mo)*30*24*3600 + (end_d-start_d)*24*3600 + (end_h-start_h)*3600 + (end_m-start_m)*60 + (end_s-start_s)
            
#             assert out_dict[ppl_key]['time'] > 0, print(end_y, end_mo, end_d, end_h, end_m, end_s, start_y, start_mo, start_d, start_h, start_m, start_s)

#             step = int(parse_step_from_log(lines[idx-2]))

#             out_dict[ppl_key]['step'] = step

#         elif 'Training time:' in l and stage_1:
#             t_minutes = re.search('Training time: ([+-]?([0-9]*[.])?[0-9]+) minutes', l).group(1)

#             out_dict['train_time'] = float(t_minutes) * 60

#     return out_dict


# def parse_results_from_experiment(exp_name: str,
#                                   dir_path: str,
#                                   file_type: Optional[str] = '.json',
#                                   verbose: Optional[bool] = True) -> Dict[str, Any]:
#     """Parses results from an experiment.

#     Args:
#         exp_name: Name of the experiment.
#         dir_path: Path to the directory holding the experiment's files.
#         file_type: Type of files to be parsed.
#         verbose: Whether to display additional logging or not.

#     Returns:
#         (Dict[str, Any]): Parsed results encoded into a dictionary.

#     """

#     if not isinstance(file_type, list):
#         file_type = [file_type]

#     results = {}

#     for j in os.listdir(dir_path):
#         j_path = os.path.join(dir_path, j)

#         if os.path.isdir(j_path):
#             results.update(parse_results_from_experiment(exp_name, j_path, file_type, verbose))

#         else:
#             for ft in file_type:
#                 if ft in j:
#                     if '.log' in ft:
#                         logs = parse_info_from_log_file(j_path, stage_1='stage_1' in exp_name)
#                     elif '.json' in ft:
#                         logs = parse_info_from_json_file(j_path)
#                     elif '.yaml' in ft:
#                         with open(os.path.join(j_path), 'r') as f:
#                             config = yaml.load(f)

#                         if config is None:
#                             json_file = os.path.join(dir_path, 'train_log.json')

#                             with open(json_file, 'r', encoding='utf-8') as f:
#                                 lines = f.readlines()

#                                 try:
#                                     job_desc = re.search('DLLL \{(.+?)\}', lines[0])
#                                 except:
#                                     return None

#                                 job_desc = '{'+job_desc.group(1)+'}}'
#                                 config = json.loads(job_desc)['data']
#                                 config['n_token'] = 267735

#                         model_config = {k: config[k] for k in KEYS_MODEL_CONFIG}

#                         cutoffs, tie_projs = [], [False]

#                         if config['adaptive']:
#                             assert config['dataset'] in ['wt103', 'wt2', 'lm1b']

#                             if config['dataset'] in ['wt103', 'wt2']:
#                                 cutoffs = [19997, 39997, 199997]
#                                 tie_projs += [True] * len(cutoffs)
#                             elif config['dataset'] == 'lm1b':
#                                 cutoffs = [59997, 99997, 639997]
#                                 tie_projs += [False] * len(cutoffs)

#                         model_config['cutoffs'] = cutoffs
#                         model_config['tie_projs'] = tie_projs
#                         model_config['tie_weight'] = config['tied']
#                         model_config['dtype'] = None
                        
#                         logs = model_config

#                     else:
#                         assert False, 'unsupported file type {}'.format(ft)

#                     if logs:
#                         config_name = parse_config_name_from_job(j_path)

#                         if verbose:
#                             print(config_name, logs)
                            
#                         if config_name in results.keys():
#                             results[config_name].update(logs)
#                         else:
#                             results[config_name] = logs

#     return results


# def parse_results_from_amulet(n_population: int,
#                               exp_name: str,
#                               results_path: str,
#                               n_trials: int,
#                               n_configs: int,
#                               start_config: int) -> Tuple[List[float], List[Dict[str, Any]]]:
#     """Parses results from an Amulet experiment.

#     Args:
#         n_population: Number of agents (genes) in the population.
#         exp_name: Name of the experiment.
#         results_path: Path to the directory holding the results' files.
#         n_trials: Number of trials.
#         n_configs: Number of configurations.
#         start_config: From which config parsing should start.

#     Returns:
#         (Tuple[List[float], List[Dict[str, Any]]]): Validation perplexities and sorted list
#             of configuration files.

#     """

#     keys = []

#     for i in range(start_config, start_config + n_configs):
#         for j in range(n_trials):
#             if len(keys) == n_population:
#                 break
#             keys.append(f'config_{i}_j{j}')

#     results = parse_results_from_experiment(exp_name, results_path, file_type='.json')

#     def _found_all_jobs(keys: List[str], results: Dict[str, Any]) -> bool:
#         for k in keys:
#             if k not in results.keys():
#                 return False
#         return True

#     while not _found_all_jobs(keys, results):
#         time.sleep(60)

#         results = parse_results_from_experiment(exp_name, results_path, file_type='.json')

#     configs = parse_results_from_experiment(exp_name, results_path, file_type='.yaml')

#     exp_results = {k: results[k] for k in keys}
#     configs_from_jobs = {k: {'d_model': configs[k]['d_model'], 'n_layer': configs[k]['n_layer'],
#                              'd_inner': configs[k]['d_inner'], 'n_head': configs[k]['n_head']} for k in keys}

#     configs_list = []
#     indices = []

#     val_ppls = np.zeros(n_population)

#     for k, v in exp_results.items():
#         config_num = int(re.search('config_([0-9]+)', k).group(1))
#         job_num = int(re.search('j([0-9]+)', k).group(1))

#         val_ppls[(config_num * n_trials) + job_num] = v['valid_perplexity']

#         configs_list.append(configs_from_jobs[k])
#         indices.append((config_num * n_trials) + job_num)

#     configs_list_sorted = []
#     for i in range(len(configs_list)):
#         idx = indices.index(i)
#         configs_list_sorted.append(configs_list[idx])

#     return val_ppls, configs_list_sorted


# def parse_results_from_baseline_experiment(args: Dict[str, Any],
#                                            exp_name: str,
#                                            dir_path: str) -> None:
#     """Parses results from a baseline experiment.

#     Args:
#         args: Additional arguments.
#         exp_name: Name of the experiment.
#         dir_path: Path to the directory holding the experiment's files.

#     """

#     baseline_results = parse_results_from_experiment(exp_name, os.path.join(dir_path, exp_name), file_type=['config.yaml', '.json'], verbose=False)

#     with open(os.path.join(dir_path, exp_name, 'latency_summary_{}.yaml'.format(args['device_name'])), 'r') as f:
#         latencies = yaml.load(f)

#     with open(os.path.join(dir_path, exp_name, 'params.pkl'), 'rb') as f:
#         params = pickle.load(f)

#     latencies_list = []
#     params_list = []
#     val_ppls = []

#     for config_name, latency in latencies.items():
#         latencies_list.append(latency)
#         params_list.append(params[config_name])
#         val_ppls.append(baseline_results[config_name]['valid_perplexity'])

#     print(f'summarized {len(latencies.keys())} baseline jobs')

#     plt.figure()
#     plt.scatter(np.asarray(latencies_list) * 1000., val_ppls, s=5)
#     plt.xlabel('Latency (ms)')
#     plt.ylabel('Val PPL')
#     plt.grid(axis='y')
#     plt.savefig(os.path.join(args['results_path'], 'baseline_pareto_latency.png'), bbox_inches="tight")

#     return latencies_list, params_list, val_ppls


def parse_values_from_yaml(value: Any) -> str:
    """Parses incoming values from a YAML file.

    Args:
        value: Incoming value.

    Returns:
        (str): String holding the parsed values.

    """

    if isinstance(value, list):
        value_string = ''
        for v in value:
            value_string += (str(v) + ' ')

        return value_string[:-1]

    else:
        return str(value)
