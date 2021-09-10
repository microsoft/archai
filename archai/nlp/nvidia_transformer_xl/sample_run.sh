#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.25
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.75

#latency-vs-params

# 8-3-exposure-bias.out
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 10 --num_prompts 500
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 20 --num_prompts 500
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 30 --num_prompts 500
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 40 --num_prompts 500
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/v1corrected/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 50 --num_prompts 500

# word level different sampling
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 1 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 10 --seed 123 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 40 --seed 123 --cuda

# char level different sampling
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 1 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 10 --seed 123 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 40 --seed 123 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 10 --seed 456 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 40 --seed 456 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 10 --seed 789 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 40 --seed 789 --cuda


# word level inference latency
#CUDA_VISIBLE_DEVICES= python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 192 --mem_len 1000 --same_length --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 1 --prompt_context_percent 0.5 --num_prompts 200 --num_chars_generate 100

# generation results
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --cuda --suggestion_length 10
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name inference --batch_size 64 --prompt_context_percent 0.5 --topk 1 --cuda --suggestion_length 10 

# exposure bias results
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M  --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 20 --topk 1 --cuda --num_prompts 5000 --suggestion_length 3 --exposure_num_prompt_tokens 10 --exposure_num_generation_tokens 10
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182 --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --cuda --model /home/t-gjawahar/object_dir/transxl-mojan-7374907516.45334-63215182/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 64 --prompt_context_percent 0.5 --prefix_len 20 --topk 1 --cuda --num_prompts 5000 --suggestion_length 3 --exposure_num_prompt_tokens 10 --exposure_num_generation_tokens 10

# 8-12
#bash /home/t-gjawahar/archai/scripts/8-12/word_accuracy.sh > /home/t-gjawahar/archai/scripts/8-12/word_accuracy.out
#bash /home/t-gjawahar/archai/scripts/8-12/char_accuracy.sh > /home/t-gjawahar/archai/scripts/8-12/char_accuracy.out
#bash /home/t-gjawahar/archai/scripts/8-12/word_latency.sh > /home/t-gjawahar/archai/scripts/8-12/word_latency.out
#bash /home/t-gjawahar/archai/scripts/8-12/char_latency.sh > /home/t-gjawahar/archai/scripts/8-12/char_latency.out
#bash /home/t-gjawahar/archai/scripts/8-12/word_memutil.sh > /home/t-gjawahar/archai/scripts/8-12/word_memutil.out
#bash /home/t-gjawahar/archai/scripts/8-12/char_pemutil.sh > /home/t-gjawahar/archai/scripts/8-12/char_pemutil.out

# 8-12-enron
#bash /home/t-gjawahar/archai/scripts/8-12-enron/word_accuracy.sh > /home/t-gjawahar/archai/scripts/8-12-enron/word_accuracy.out
#bash /home/t-gjawahar/archai/scripts/8-12-enron/char_accuracy.sh > /home/t-gjawahar/archai/scripts/8-12-enron/char_accuracy.out

# 8-16 office dataset
# wiki models
# bash /home/t-gjawahar/archai/scripts/8-12/word_accuracy.sh > /home/t-gjawahar/archai/scripts/8-16/word_accuracy.out
# bash /home/t-gjawahar/archai/scripts/8-12/char_accuracy.sh > /home/t-gjawahar/archai/scripts/8-16/char_accuracy.out
# reddit models
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/word_train/word80M_nofp_reddit_g4 --dataset wt2 --tgt_len 192 --mem_len 192 --same_length --model /home/t-gjawahar/archai/amlt/word_train/word80M_nofp_reddit_g4/checkpoint_best.pt --experiment_name word_accuracy --batch_size 128 --prompt_context_percent 0.5 --cuda
#python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/word_train/char80M_nofp_reddit_g4 --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/word_train/char80M_nofp_reddit_g4/checkpoint_best.pt --experiment_name char_accuracy --batch_size 128 --prompt_context_percent 0.5 --cuda

# tokenization
<<comment
for vsize in 128 50000 1000 10000; do
  export VOCAB_SIZE="$vsize"
  export TOKENIZER_DIR=TokS"$vsize"
  mkdir -p TokS"$vsize"
  #python /home/t-gjawahar/archai/archai/nlp/nlxpy/nlxpy/cli/sk_bbpe_tokenize.py --verbose --input train.txt --add_prefix_space --add_prefix_new_line --min_frequency 1000 --tokenizer_dir $TOKENIZER_DIR --special_tokens "_OOV_,_BOS_" --vocab_size $VOCAB_SIZE --sort --train --output $TOKENIZER_DIR/train.txt.tok
  python /home/t-gjawahar/archai/archai/nlp/nlxpy/nlxpy/cli/sk_bbpe_tokenize.py --verbose --input train.txt --add_prefix_space --add_prefix_new_line --min_frequency 1000 --tokenizer_dir $TOKENIZER_DIR --output $TOKENIZER_DIR/train.txt.tok
  #python /home/t-gjawahar/archai/archai/nlp/nlxpy/nlxpy/cli/sk_bbpe_tokenize.py --verbose --input valid.txt --add_prefix_space --add_prefix_new_line --min_frequency 1000 --tokenizer_dir $TOKENIZER_DIR --output $TOKENIZER_DIR/valid.txt.tok
  #python /home/t-gjawahar/archai/archai/nlp/nlxpy/nlxpy/cli/sk_bbpe_tokenize.py --verbose --input test.txt  --add_prefix_space --add_prefix_new_line --min_frequency 1000 --tokenizer_dir $TOKENIZER_DIR --output $TOKENIZER_DIR/test.txt.tok
done
comment
