# python archai/nlp/search.py --n_iter 30 --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp

# python archai/nlp/search.py --profile_baseline --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp

python archai/nlp/search.py --select_pareto --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
        --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp