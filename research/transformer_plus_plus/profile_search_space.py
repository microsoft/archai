from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool

import json
from tqdm import tqdm
from archai.nlp.eval.profiler import profile
from transformers import AutoTokenizer, GPT2Config

from transformer_plus_plus.search_space.search_space import (
    build_single_op_ss,
    build_single_op_per_layer_ss,
    build_mixed_attention_hom_ss,
    build_mixed_attention_het_ss
)

SS_TYPES = {
    'single_op': build_single_op_ss,
    'single_op_layer': build_single_op_per_layer_ss,
    'mixed_attention_hom': build_mixed_attention_hom_ss,
    'mixed_attention_het': build_mixed_attention_het_ss
}

parser = ArgumentParser()
parser.add_argument('-o', '--output', type=Path, required=True)
parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2'])
parser.add_argument('--ss', type=str, default='gpt2', choices=list(SS_TYPES.keys()))
parser.add_argument('--seq_len', type=int, default=2048)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--nb_measurements', type=int, default=3)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--nb_proc', type=int, default=4, help='Number of processes to use for sampling models')
parser.add_argument('--nb_layers', type=int, default=4)


if __name__ == '__main__':
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sample_input = tokenizer(['test' * args.seq_len], return_tensors='pt').to(args.device)

    def _sample_models(sample_size, seed):
        search_space = SS_TYPES[args.ss](hf_config=GPT2Config(n_positions=args.seq_len), seed=seed,
                                         min_layers=args.nb_layers, max_layers=args.nb_layers)
        return [search_space.random_sample() for _ in tqdm(range(sample_size))]

    print(f'Sampling {args.num_samples} models from {args.ss} search space')
    with Pool(args.nb_proc) as pool:
        models = pool.starmap(
            _sample_models,
            [(args.num_samples // args.nb_proc, seed) for seed in range(args.nb_proc)]
        )

    # Remove duplicates
    print('Removing duplicates')
    models = [model for model_list in models for model in model_list]
    models = list({model.archid: model for model in models}.values())

    results = []
    for model in tqdm(models, desc='Profiling models'):
        try:
            model.arch.to(args.device)

            model_results = [
                profile(
                    model.arch, model_kwargs=sample_input, n_warmups=args.warmup
                ) for _ in range(args.nb_measurements)
            ]

            model_results = {
                k: sum([r[k] for r in model_results]) / args.nb_measurements
                for k in model_results[0].keys()
            }

            model_results['archid'] = model.archid
            model_results['arch_config'] = model.metadata['config'].to_dict()

            results.append(model_results)

            # reconverting to cpu to avoid memory leak
            model.arch.to('cpu')
        except Exception as e:
            print(f'Error profiling model {model.archid}: {e}')
            continue
    
    json.dump(results, open(args.output / f'{args.ss}_{args.seq_len}.json', 'w'), indent=2)
