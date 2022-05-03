from archai.nlp.datasets.lm_iterators import LMOrderedIterator
from synflow_utils import compute_synflow_per_weight
from archai.nlp.nas.zero_cost_utils.flops import get_model_flops
import torch
import os
import numpy as np
import collections
import yaml
import collections
import argparse
import re
import types
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from archai.nlp.models.model_loader import load_model_from_config

plt.rcParams.update({"font.size": 18})


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value


def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar",
    meta_constructor_sequence,
)
yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy.dtype", meta_constructor_mapping
)


def forward_synflow(
    self, data, target, mems, output_loss=True, output_prediction_scores=False
):
    # Causal attention mask is created inside the model
    outputs = self.model(
        input_ids=data,
        labels=target,
        attention_mask=torch.ones_like(data),
        output_loss=output_loss,
        output_prediction_scores=output_prediction_scores,
    )

    return outputs


def get_scores(args, exp_name):
    path_to_results = exp_name
    yaml_file_scores = os.path.join(
        path_to_results, "synflow_scores_seed_{}.yaml".format(args.seed)
    )
    yaml_file_cost = os.path.join(path_to_results, "synflow_cost.yaml")

    calc_scores = True
    calc_costs = False

    device = torch.device("cpu")

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

    if calc_scores or calc_costs:
        scores = {}
        costs = {}
        count = 1
        for _f in set(files):
            if "model_config.yaml" in _f:
                config_name = "model_config.yaml"
                with open(_f, "r") as f:
                    model_config = yaml.full_load(f)

                model = load_model_from_config("hf_gpt2_flex", model_config)
                model.n_token = model_config["n_token"]

                model.forward = types.MethodType(forward_synflow, model)

                # print(model)

                # curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model)
                # nparams[config_name] = {'AdaEmb': float(params_adaptive_embedding), 'Sftmax': float(params_adaptive_softmax), \
                #                         'Attn': float(params_attention), 'FFN': float(params_ff), 'total': float(curr_n_all_param)}

                B = 1
                tgt_len, mem_len, ext_len = (
                    model_config["tgt_len"],
                    model_config["mem_len"],
                    model_config["ext_len"],
                )
                data_len = tgt_len
                data = torch.ones(data_len * B).to(device, torch.long)
                diter = LMOrderedIterator(
                    data, B, tgt_len, device=device, ext_len=ext_len
                )
                if calc_scores:
                    for idx, (inp, tgt, seqlen, _) in enumerate(diter):
                        grads_abs = compute_synflow_per_weight(model, inp, tgt)
                        score = np.sum(
                            [torch.sum(g).detach().numpy() for g in grads_abs]
                        )
                        break
                    scores[config_name] = score.tolist()
                    print(count, config_name, scores[config_name])

                if calc_costs:
                    model.eval()
                    with torch.no_grad():
                        for _, (inp, tgt, _, _) in enumerate(diter):
                            curr_flops = get_model_flops(model, inp, tgt)
                            total_flops = np.sum(
                                [curr_flops[k] for k in ["Attn", "FFN", "Sftmax"]]
                            ).tolist()
                            break
                    costs[config_name] = 3 * total_flops
                    print(count, config_name, costs[config_name])

    if calc_scores:
        with open(yaml_file_scores, "w") as f:
            yaml.dump(scores, f)
    if calc_costs:
        with open(yaml_file_cost, "w") as f:
            yaml.dump(costs, f)
    # with open(os.path.join(path_to_results, 'synflow_params.yaml'), 'w') as f:
    #   yaml.dump(nparams, f)


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


def get_statistics(seed, results_gt, scores, nparams_dict, topk_list):
    common_configs = np.intersect1d(list(results_gt.keys()), list(scores[seed].keys()))
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
        target_scores.append(-scores[seed][k])
    sorted_target = np.argsort(target_scores)

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


def plot(args):
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

        scores = {}
        for file in os.listdir(path_to_results):
            if "synflow_scores_seed" in file:
                if "old" in file or "decoderOnly" in file or "nclusters" in file:
                    continue
                seed = re.search("seed_([0-9]+)", file).group(1)
                with open(os.path.join(path_to_results, file), "r") as f:
                    print("loading scores for seed {} from {}".format(seed, file))
                    scores[seed] = yaml.safe_load(f)

        common_ratios[legend_key] = {}
        spr_ranks[legend_key] = {}
        param_corrs[legend_key] = {}
        topk_list = range(10, 101, 10)
        if args.cross_seed:
            for seed in scores.keys():
                common_ratio, spr_rank, param_corr = get_statistics(
                    seed, results_gt, scores, nparams_dict, topk_list
                )
                common_ratios[legend_key][seed] = common_ratio
                spr_ranks[legend_key][seed] = spr_rank
                param_corrs[legend_key][seed] = param_corr
        else:
            common_ratio, spr_rank, param_corr = get_statistics(
                str(args.seed), results_gt, scores, nparams_dict, topk_list
            )
            common_ratios[legend_key][str(args.seed)] = common_ratio
            spr_ranks[legend_key][str(args.seed)] = spr_rank
            param_corrs[legend_key][str(args.seed)] = param_corr

    plt.figure()
    param_types = list(param_corr.keys())
    for lk in legend_keys:
        for seed in common_ratios[lk].keys():
            corrs = [param_corrs[lk][seed][pt] for pt in param_types]
            print(corrs)
            plt.scatter(
                range(1, len(param_types) + 1), corrs, label=lk + "_seed_" + seed
            )
    plt.xticks(range(1, len(param_types) + 1), list(param_types))
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(axis="y")
    plt.title("Synflow score correlation with nparams")
    plt.savefig("synflow_params.png", bbox_inches="tight")

    plt.figure()
    for lk in legend_keys:
        for seed in common_ratios[lk].keys():
            plt.scatter(topk_list, common_ratios[lk][seed], label=lk + "_seed_" + seed)
    plt.ylabel("Common ratio")
    plt.xlabel("Topk (%)")
    plt.xticks(topk_list)
    plt.title("ranking based on Synflow")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(axis="y")
    plt.savefig("common_ratio_topk_synflow.png", bbox_inches="tight")

    plt.figure()
    for lk in legend_keys:
        for seed in common_ratios[lk].keys():
            plt.scatter(topk_list, spr_ranks[lk][seed], label=lk + "_seed_" + seed)
    plt.ylabel("Spearman's Correlation")
    plt.xlabel("Topk (%)")
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.grid(axis="y")
    plt.title("ranking based on Synflow")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("spearman_topk_synflow.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results Analysis.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="/home/gderosa/archai/amlt/",
        help="path to the experiment",
    )
    parser.add_argument("--seed", type=int, default=1111, help="Random seed")
    parser.add_argument(
        "--plot", action="store_true", help="plot the spearman corr and common ratio"
    )
    parser.add_argument(
        "--cross_seed",
        action="store_true",
        help="plot the spearman corr and common ratio for all evaluated seeds",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    get_scores(args, args.exp_name)

    if args.plot:
        plot(args)
