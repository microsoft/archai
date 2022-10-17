set -e -o xtrace

bash scripts/dist_main.sh --full --no-search --algos darts --datasets cifar10 --nas.eval.final_desc_filename models/darts/darts_genotype.yaml --common.apex.min_world_size 2 --nas.eval.trainer.apex.enabled True