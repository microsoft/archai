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

from archai.nlp.models.model_loader import load_model_from_config

plt.rcParams.update({"font.size": 18})


def get_metrics(
    topk,
    sorted_ground_truth,
    sorted_target,
    val_ppl_list_gt,
    val_ppl_list_target,
    common_configs=None,
):
    idx = int(topk / 100.0 * len(sorted_ground_truth))
    sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
    sorted_target_binned = sorted_target[:idx].astype(np.int32)

    correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
    total = len(sorted_target_binned)
    common_ratio = correct * 1.0 / total
    print(
        "Correctly ranked top %d %% (%d) with %.2f accuracy"
        % (topk, total, correct * 1.0 / total)
    )

    topk_val_ppl_list_gt = [
        val_ppl_list_gt[i]
        for i in range(len(val_ppl_list_gt))
        if i in sorted_ground_truth_binned
    ]
    topk_val_ppl_list_target = [
        val_ppl_list_target[i]
        for i in range(len(val_ppl_list_target))
        if i in sorted_ground_truth_binned
    ]
    # topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_target_binned]
    # topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_target_binned]
    spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)
    print(
        "Spearman Correlation on top %d %% (%d): %.3f"
        % (topk, len(topk_val_ppl_list_gt), spr_rank)
    )
    # kendal_tau, _ = kendalltau(topk_val_ppl_list_gt, topk_val_ppl_list_target)
    # print('Kendal tau on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), kendal_tau))

    return common_ratio, spr_rank


def get_scores(args, exp_name, tr_iter, method="snip", compute_cost=False):
    path_to_results = exp_name
    yaml_file = os.path.join(path_to_results, f"{method}_scores_seed_{args.seed}.yaml")

    scores = {}
    costs = {}
    files = []
    dirlist = [path_to_results]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
            files.extend(
                map(
                    lambda n: os.path.join(*n),
                    zip([dirpath] * len(filenames), filenames),
                )
            )

    count = 1
    for _f in set(files):
        if "model_config.yaml" in _f:
            config_name = "model_config.yaml"
            with open(_f, "r") as f:
                model_config = yaml.full_load(f)

            model = load_model_from_config("hf_gpt2_flex", model_config)
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
                print(count, config_name, scores[config_name], costs[config_name])
            else:
                print(count, config_name, scores[config_name])
            count += 1

    with open(yaml_file, "w") as f:
        yaml.dump(scores, f)

    if compute_cost:
        with open(os.path.join(path_to_results, f"{method}_cost.yaml"), "w") as f:
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
        val_ppl_list_gt.append(results_gt[k]["valid_perplexity"])
    sorted_ground_truth = np.argsort(val_ppl_list_gt)

    # zero-cost score results:
    target_scores = []
    for k in common_configs:
        # *1./param_count)   # the higher the score, the better the architecture (reversely correlated with ppl)
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
    legend_keys = []

    for exp_name in args.exp_name:
        path_to_results = os.path.join(args.results_dir, exp_name)
        legend_key = "heterogeneous" if "heterogeneous" in exp_name else "homogeneous"
        legend_keys.append(legend_key)

        # load the ground-truth rankings
        yaml_file = os.path.join(path_to_results, "result_summary.yaml")
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

        common_ratios[legend_key] = {}
        spr_ranks[legend_key] = {}
        param_corrs[legend_key] = {}
        topk_list = [10, 30, 50, 100]  # range(10,101,10)
        for m in scores.keys():
            print(f"------------ {m} ------------")
            if m == "grasp":
                prev_scores = scores[m]
                scores[m] = {k: -s for k, s in prev_scores.items()}
            common_ratio, spr_rank, param_corr = get_statistics(
                m, results_gt, scores, nparams_dict, topk_list
            )
            common_ratios[legend_key][m] = common_ratio
            spr_ranks[legend_key][m] = spr_rank
            param_corrs[legend_key][m] = param_corr

    plt.figure()
    param_types = list(param_corr.keys())
    for lk in legend_keys:
        for i, m in enumerate(common_ratios[lk].keys()):
            corrs = [param_corrs[lk][m][pt] for pt in param_types]
            plt.scatter(
                range(1, len(param_types) + 1), corrs, label=m
            )  # lk+'_method_'+m)
    plt.xticks(range(1, len(param_types) + 1), list(param_types))
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(axis="y")
    plt.title("zero-cost score correlation with nparams")
    plt.savefig("zero-cost_params.png", bbox_inches="tight")

    plt.figure()
    for lk in legend_keys:
        for m in common_ratios[lk].keys():
            # lk+'_method_'+m)
            plt.scatter(topk_list, common_ratios[lk][m], label=m)
    plt.ylabel("Common ratio")
    plt.xlabel("Topk (%)")
    plt.xticks(topk_list)
    plt.title("ranking based on zero-cost methods")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(axis="y")
    plt.savefig(f"common_ratio_topk_zero-cost_{exp_name}.png", bbox_inches="tight")

    plt.figure()
    for lk in legend_keys:
        for i, m in enumerate(common_ratios[lk].keys()):
            # lk+'_method_'+m)
            plt.scatter(topk_list, spr_ranks[lk][m], label=m)
    plt.ylabel("Spearman's Correlation")
    plt.xlabel("Topk (%)")
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.grid(axis="y")
    plt.title("ranking based on zero-cost methods")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"spearman_topk_zero-cost_{exp_name}.png", bbox_inches="tight")

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
    for lk in legend_keys:
        for i, m in enumerate(common_ratios[lk].keys()):
            l = labels[m]
            avg_cost = np.mean(list(costs[m].values()))
            # lk+'_method_'+m)
            plt.scatter(
                avg_cost,
                spr_ranks[lk][m][-1],
                label=l,
                marker=markers[i],
                s=180,
                c=colors[i],
            )
    plt.ylabel("Spearman's Correlation")
    plt.xlabel("FLOPs Cost")
    plt.ylim((0.8, 1.0))
    plt.grid(axis="y")
    # plt.title('ranking based on zero-cost methods')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"spearman_cost_zero-cost_{exp_name}.png", bbox_inches="tight")


def cost_fn(method, model, tr_iter, device):
    model.eval()
    with torch.no_grad():

        for _, (inp, tgt, _, _) in enumerate(tr_iter):
            inp.to(device)
            tgt.to(device)
            curr_flops = get_model_flops(model, inp, tgt)
            total_flops = np.sum([curr_flops[k] for k in curr_flops.keys()]).tolist()
            break

    cost = 0
    if method in ["fisher", "grasp"]:
        n_linear_layers = 0
        found_compute_cost = 0
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, transformers.Conv1D):
                n_linear_layers += 1
            if hasattr(layer, "compute"):
                cost += layer.compute
                found_compute_cost += 1
            assert (
                found_compute_cost == n_linear_layers
            ), f"{found_compute_cost}, {n_linear_layers}"
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
    parser.add_argument(
        "--exp_name",
        type=str,
        default="/home/gderosa/archai/amlt/",
        help="path to the experiment",
    )
    parser.add_argument(
        "--get_cost", action="store_true", help="compute cost for each method"
    )
    parser.add_argument("--seed", type=int, default=1111, help="Random seed")
    parser.add_argument(
        "--plot", action="store_true", help="plot the spearman corr and common ratio"
    )
    parser.add_argument(
        "--method", type=str, default="snip", help="zero-cost method to use"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="use gpu for score calculation"
    )

    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wt103",
        choices=["wt103", "lm1b"],
        help="Dataset name",
    )
    parser.add_argument(
        "--vocab", type=str, default="gpt2", choices=["gpt2"], help="Type of vocabulary"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=None, help="Size of vocabulary"
    )
    parser.add_argument(
        "--dataload", type=str, default="random", help="random or grasp supported"
    )
    parser.add_argument(
        "--dataload_info",
        type=int,
        default=1,
        help="number of batches to use for random dataload or number of samples per class for grasp dataload",
    )
    parser.add_argument("--start", type=int, default=5, help="start index")
    parser.add_argument("--end", type=int, default=10, help="end index")

    parser.add_argument(
        "--write_freq", type=int, default=100, help="frequency of write to file"
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    device = "cuda"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "wt103":
        eval_batch_size = 8
        eval_tgt_len = 192
    elif args.dataset == "lm1b":
        eval_batch_size = 8
        eval_tgt_len = 32

    data, work_dir, _, cache_dir, dataroot = exp_utils.get_create_dirs(
        None, args.dataset, pretrained_path="", cache_dir="cache"
    )

    corpus = get_lm_corpus(
        data, cache_dir, args.dataset, "gpt2", vocab_size=50257, refresh_cache=False
    )
    train_itr = corpus.get_iterator(
        "train", eval_batch_size, eval_tgt_len, device=device, mem_len=0, ext_len=0
    )

    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    methods = ["snip", "grad_norm", "fisher", "jacob_cov", "grasp", "jacob_cov_relu"]
    for method in methods:
        get_scores(
            args, args.exp_name, train_itr, method=method, compute_cost=args.get_cost
        )

    if args.plot:
        plot(args, methods=methods + ["synflow"])
