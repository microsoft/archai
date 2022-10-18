import transformers
from pruners import predictive
from archai.nlp.datasets.distributed_utils.data_utils import get_lm_corpus
from archai.nlp.datasets import exp_utils
from archai.nlp.nas.zero_cost_utils.flops import get_model_flops
import torch.nn as nn
import torch
import os
import numpy as np
import collections
import yaml
import collections
import argparse
import re
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from archai.nlp.nas.zero_cost_utils.pruners.measures.synflow import get_synflow_scores
from archai.nlp.models.model_loader import load_model_from_config

plt.rcParams.update({"font.size": 18})


def get_metrics(topk, sorted_ground_truth, sorted_target, val_ppl_list_gt, val_ppl_list_target, common_configs=None):
    idx = int(topk / 100.0 * len(sorted_ground_truth))
    sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
    sorted_target_binned = sorted_target[:idx].astype(np.int32)

    correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
    total = len(sorted_target_binned)
    common_ratio = correct * 1.0 / total
    print("Correctly ranked top %d %% (%d) with %.2f accuracy" % (topk, total, correct * 1.0 / total))

    topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_ground_truth_binned]
    topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_ground_truth_binned]
    # topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_target_binned]
    # topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_target_binned]
    spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)
    print("Spearman Correlation on top %d %% (%d): %.3f" % (topk, len(topk_val_ppl_list_gt), spr_rank))
    # kendal_tau, _ = kendalltau(topk_val_ppl_list_gt, topk_val_ppl_list_target)
    # print('Kendal tau on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), kendal_tau))

    return common_ratio, spr_rank


def get_scores(args, exp_name, tr_iter, method="snip", compute_cost=False):
    path_to_results = exp_name
    
    scores = {}
    costs = {}
    files = []
    dirlist = [path_to_results]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
            files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames),))

    count = 1
    yaml_file = os.path.join(path_to_results, f"{method}_scores_seed_{args.seed}.yaml")
    cost_file = os.path.join(path_to_results, f"{method}_cost.yaml")
    if not os.path.exists(yaml_file) or (compute_cost and not os.path.exists(cost_file)):
        for _f in set(files):
            if "model_config.yaml" in _f:
                idx =  re.search('(config_[0-9]+)', _f).span()[0]
                job = _f[idx:]
                config_name = job.split('/')[0] + '_' + job.split('/')[1]
                with open(_f, "r") as f:
                    model_config = yaml.full_load(f)
                
                model = load_model_from_config(args.model_type, model_config)
                model.n_token = model_config["n_token"]
                measures = predictive.find_measures(
                    args,
                    model,
                    tr_iter,
                    (args.dataload, args.dataload_info, args.n_token),
                    args.device,
                    measure_names=[method],
                )
                scores[config_name] = measures[method]
                if compute_cost:
                    cost = cost_fn(method, model, tr_iter, args.device)
                    costs[config_name] = cost
                    print(count, config_name, 'score:', scores[config_name], 'FLOPS:', costs[config_name])
                else:
                    print(count, config_name, 'score:', scores[config_name])
                count += 1
        
        print(f'saving to {yaml_file}')
        with open(yaml_file, "w") as f:
            yaml.dump(scores, f)

        if compute_cost:
            with open(cost_file, "w") as f:
                yaml.dump(costs, f)


def get_statistics(method, results_gt, scores, nparams_dict, topk_list):
    old_keys = list(scores[method].keys())
    for k in old_keys:
        if "_config" in k:
            idx = re.search("(_config)", k).span()[0]
            new_key = k[:idx]
            scores[method][new_key] = scores[method][k]
            del scores[method][k]

    common_configs = np.intersect1d(
        list(results_gt.keys()), list(scores[method].keys())
    )
    print("analyzing {} architectures".format(len(common_configs)))

    # fear_stage_1 results:
    val_ppl_list_gt = []
    for k in common_configs:
        val_ppl_list_gt.append(results_gt[k]["valid_ppl"])
    sorted_ground_truth = np.argsort(val_ppl_list_gt)

    # zero-cost score results:
    target_scores = []
    for k in common_configs:
        target_scores.append(-scores[method][k])
    sorted_target = np.argsort(target_scores)

    fig, ax1 = plt.subplots()
    ax1.plot(
        range(len(val_ppl_list_gt)),
        np.asarray(val_ppl_list_gt)[sorted_ground_truth],
        label="val_ppl",
    )
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(val_ppl_list_gt)),
        np.asarray(target_scores)[sorted_ground_truth],
        label="# decoder params",
    )
    plt.legend()
    plt.savefig("validation_ppls.png")

    # parameters
    nparams = {}
    for k in common_configs:
        for param_type in nparams_dict[k].keys():
            try:
                nparams[param_type].append(nparams_dict[k][param_type])
            except:
                nparams[param_type] = [nparams_dict[k][param_type]]
    param_corr = {}
    for param_type, target_params in nparams.items():
        param_corr[param_type], _ = spearmanr(
            (-np.asarray(target_scores)).tolist(), target_params
        )

    common_ratios = []
    spr_ranks = []
    # extract common ratio and spearmanrank
    for topk in topk_list:
        common_ratio, spr_rank = get_metrics(
            topk,
            sorted_ground_truth=sorted_ground_truth,
            sorted_target=sorted_target,
            val_ppl_list_gt=val_ppl_list_gt,
            val_ppl_list_target=target_scores,
        )
        common_ratios.append(common_ratio)
        spr_ranks.append(spr_rank)

    return common_ratios, spr_ranks, param_corr


def plot(args, methods):
    common_ratios = {}
    spr_ranks = {}
    param_corrs = {}

    path_to_results = args.exp_name
    
    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, "ppl_summary.yaml")
    with open(yaml_file, "r") as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))

    with open(os.path.join(path_to_results, "params_summary.yaml"), "r") as f:
        nparams_dict = collections.OrderedDict(yaml.safe_load(f))

    costs = {}
    scores = {}
    for m in methods:
        fname = f"{m}_scores_seed_{args.seed}.yaml"
        with open(os.path.join(path_to_results, fname), "r") as f:
            print("loading scores for method ", m)
            scores[m] = yaml.safe_load(f)
        fname = f"{m}_cost.yaml"
        with open(os.path.join(path_to_results, fname), "r") as f:
            print("loading costs for method ", m)
            costs[m] = yaml.safe_load(f)

    costs["nparams"] = {}
    scores["nparams"] = {}
    for k, v in nparams_dict.items():
        scores["nparams"][k] = v["Attn"] + v["FFN"]
        costs["nparams"][k] = 0.0

    topk_list = [10, 30, 50, 100]  # range(10,101,10)
    for m in scores.keys():
        print(f"------------ {m} ------------")
        if m == "grasp":
            prev_scores = scores[m]
            scores[m] = {k: -s for k, s in prev_scores.items()}
        common_ratio, spr_rank, param_corr = get_statistics(m, results_gt, scores, nparams_dict, topk_list)
        print('avg FLOPs:', np.mean(list(costs[m].values())))
        common_ratios[m] = common_ratio
        spr_ranks[m] = spr_rank
        param_corrs[m] = param_corr

    markers = ["o", "^", "s", "P", "*", "X", "d", "v"]
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(markers)))
    labels = {
        "grad_norm": "GradNorm",
        "snip": "Snip",
        "fisher": "Fisher",
        "jacob_cov": "JacobCov",
        "grasp": "Grasp",
        "jacob_cov_relu": "ReLU",
        "synflow": "Synflow",
        "nparams": "# Params",
    }
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(common_ratios.keys()):
        l = labels[m]
        avg_cost = np.mean(list(costs[m].values()))
        plt.scatter(
            avg_cost,
            spr_ranks[m][-1],
            label=l,
            marker=markers[i],
            s=180,
            c=colors[i],
        )
    plt.ylabel("Spearman's Correlation")
    plt.xlabel("FLOPs Cost")
    plt.ylim((0.79, 1.0))
    # plt.xlim((-1e10, 2e11))
    plt.grid(axis="y")
    # plt.title('ranking based on zero-cost methods')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    path_to_plot = os.path.join(args.exp_name,'plots')
    os.makedirs(path_to_plot, exist_ok=True)
    plt.savefig(os.path.join(path_to_plot, f"spearman_cost_zero-cost.png", bbox_inches="tight")


def cost_fn(method, model, tr_iter, device):
    model.eval()
    with torch.no_grad():
        for _, (inp, tgt, _, _) in enumerate(tr_iter):
            curr_flops = get_model_flops(model, inp, tgt)
            # print(curr_flops)
            total_flops = np.sum([curr_flops[k] for k in curr_flops.keys()]).tolist()
            break

    cost = 0
    if method in ["fisher", "grasp"]:
        n_linear_layers = 0
        found_compute_cost = 0
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                n_linear_layers += 1
            if hasattr(layer, "compute"):
                cost += layer.compute
                found_compute_cost += 1
                # print(found_compute_cost, layer.compute)
        assert (found_compute_cost == n_linear_layers), f"{found_compute_cost}, {n_linear_layers}"
        assert cost > 0
        # print('compute cost:', cost)
        if method == "fisher":
            cost += 3 * total_flops
        else:
            cost += 8 * total_flops
    elif method in ["snip", "grad_norm", "jacob_cov"]:
        cost = 3 * total_flops
    elif method == "jacob_cov_relu":
        cost = total_flops + model.compute
        # n_relus = 0
        # found_compute_cost = 0
        # for layer in model.modules():
        #   if 'ReLU' in str(type(layer)):
        #     n_relus += 1
        #   if hasattr(layer, 'compute'):
        #     cost += layer.compute
        #     found_compute_cost += 1
        # assert found_compute_cost == n_relus, f'{found_compute_cost}, {n_relus}'
    else:
        raise NotImplementedError

    return cost


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-cost Metrics for Transformers")
    parser.add_argument( "--exp_name", type=str, default=".", 
                        help="path to the experiment",)
    parser.add_argument('--model_type', default='mem_transformer', type=str,
                     choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                     help='Which model type to use')
    parser.add_argument("--get_cost", action="store_true", help="compute cost for each method")
    parser.add_argument("--seed", type=int, default=1111, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="plot the spearman corr and common ratio")
    parser.add_argument("--method", type=str, default="snip", help="zero-cost method to use")
    parser.add_argument("--cuda", action="store_true", help="use gpu for score calculation")

    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--dataset", type=str, default="wt103", choices=["wt103", "lm1b"], help="Dataset name",)
    parser.add_argument("--vocab", type=str, default="gpt2", choices=["gpt2"], help="Type of vocabulary")
    parser.add_argument("--vocab_size", type=int, default=None, help="Size of vocabulary")
    parser.add_argument("--dataload", type=str, default="random", help="random or grasp supported")
    parser.add_argument("--dataload_info", type=int, default=1, 
                        help="number of batches to use for random dataload or number of samples per class for grasp dataload",)
    parser.add_argument("--start", type=int, default=5, help="start index")
    parser.add_argument("--end", type=int, default=10, help="end index")
    parser.add_argument("--write_freq", type=int, default=100, help="frequency of write to file")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


if __name__ == "__main__":
    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "wt103":
        eval_batch_size = 16
        eval_tgt_len = 192
    elif args.dataset == "lm1b":
        eval_batch_size = 16
        eval_tgt_len = 32

    data, work_dir, _, cache_dir, dataroot = exp_utils.get_create_dirs(None, args.dataset, pretrained_path="", cache_dir="cache")
    vocab = 'gpt2' if 'gpt' in args.model_type else 'word'
    vocab_size = 50257 if 'gpt' in args.model_type else 267736
    corpus = get_lm_corpus(data, cache_dir, args.dataset, vocab, vocab_size, refresh_cache=False)
    train_itr = corpus.get_iterator("train", eval_batch_size, eval_tgt_len, device=args.device, mem_len=0, ext_len=0)
    args.n_token = len(corpus.vocab)

    methods = ["snip", "grad_norm", "fisher", "jacob_cov", "grasp", "jacob_cov_relu", "synflow"]
    for method in methods:
        print(f"------------ {method} ------------")
        if method == 'synflow':
            get_synflow_scores(args, args.exp_name)
        else:
            get_scores(args, args.exp_name, train_itr, method=method, compute_cost=args.get_cost)

    if args.plot:
        plot(args, methods=methods)