pushd ~/GitHubSrc/archai

if (( $# != 1 ))
then
  echo "Usage: $0 <experiment_name>"
  exit 1
fi

python -m torch.distributed.launch --nproc_per_node="4" archai/nlp/nvidia_transformer_xl/train.py --config dgxa100_4gpu_fp16 --config_file wt103_base.yaml --experiment_name $1

popd