## Efficient Character Language Models for AutoComplete

### Training

#### vanilla character model
```
python archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer 16 --n_head 8 --d_head 64 --d_embed 750 --d_inner 2048 --mem_len 512 --tgt_len 512 --d_model 750 --dropout 0.1 -dropatt 0.0 --config dgx1_8gpu_fp16 --experiment_name char80M --config_file char_base.yaml --eval_tgt_len 1024 --batch_size 64 --lr 0.001
```
* `--config_file char_base.yaml` - default config for character model

#### character model modifications to add word/subword information
```
python archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer 16 --n_head 8 --d_head 64 --d_embed 750 --d_inner 2048 --mem_len 512 --tgt_len 512 --d_model 750 --dropout 0.1 -dropatt 0.0 --config dgx1_8gpu_fp16 --experiment_name char80M_bertstyle_word --config_file char_base.yaml --eval_tgt_len 1024 --batch_size 64 --lr 0.001 --model_ext bert_style_word_segment --segment_type word
```
* `--model_ext bert_style_word_segment` - type of modification: `bert_style_word_segment` (introduces BERT-style word/subword embedding layer), `char_emb_from_word` (pooling embeddings from characters seen so far for the current word) (default: None)
* `--segment_type word` - type of BERT-style word segment embedding (word or subword)
* `--char_pooling mean` - type of character pooling: `mean`, `max`, `sum`

#### initializing decoder layer and embeddings from another model
```
python archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer 16 --n_head 8 --d_head 64 --d_embed 750 --d_inner 2048 --mem_len 512 --tgt_len 512 --d_model 750 --dropout 0.1 -dropatt 0.0 --config dgx1_8gpu_fp16 --experiment_name char80M_layer_init --config_file char_base.yaml --eval_tgt_len 1024 --batch_size 64 --lr 0.001 --layer_init_from_ckpt /home/t-gjawahar/logdir/word80M/checkpoint_best.pt --layer_idx_to_init 0-100
```
* `--layer_init_from_ckpt /home/t-gjawahar/logdir/word80M/checkpoint_best.pt` - path to the model from which we're extracting the decoder layer weights (default: None)
* `--layer_idx_to_init 0-100` - starting and ending percent of layers to extract (e.g., `0-100` extracts all layers, `0-20` extracts bottom 20 percent layers, `80-100` extracts top 20 percent layers, `0-0` extracts no layers, which is equivalent to training with random initialization as long as `--embed_layer_init` is None. used only when `--layer_init_from_ckpt` is not None)
* `--embed_layer_init gpt2` - name of the HuggingFace model to extract input embeddings (default: None)

### Generation

#### compute exact match
```
python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/logdir/char20M --tgt_len 512 --mem_len 192 --same_length --model /home/t-gjawahar/logdir/char20M/checkpoint_best.pt --experiment_name inference_metrics --prompt_context_percent 0.2 --cuda --num_prompts 50000 --split valid --generation_method greedy
```
* `--prompt_context_percent 0.2` - percentage of words from a input document (single line in wikitext-103) to be considered for creating prompt
* `--generation_method greedy` - decoding method (`greedy` for greedy search, `beam` for beam search)
* `--beam_size 40` - beam size for the beam search (used only when `--generation_method greedy`)
* `--suggestion_length 3` - maximum length of suggestion in terms of words (default: 3)

### ONNX Export
```
python archai/nlp/nvidia_transformer_xl/onnx/onnx_export.py --checkpoint /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt --output_dir /home/t-gjawahar/archai/amlt/onnx_export
```
* `--checkpoint /home/t-gjawahar/logs/transxl_char_params_80M/checkpoint_best.pt` - checkpoint of the model to be exported
* `--output_dir /home/t-gjawahar/archai/amlt/onnx_export` - directory to store onnx checkpoints
