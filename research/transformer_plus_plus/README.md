# Transformer++

## Installation
```shell
conda create -n tfpp python=3.8
conda activate tfpp

# develop install
pip install -e .
```

## Importing an architecture
```python3

from archai.discrete_search.search_spaces.config import ArchConfig
from transformer_plus_plus.search_space.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

arch_config = ArchConfig.from_json('confs/gpt2_base/gpt2_base.json')
hf_config = GPT2Config(n_positions=1024)

model = GPT2LMHeadModel(arch_config, hf_config)

model.forward(**{
    'input_ids': torch.tensor([[1, 2, 3]]),
    'attention_mask': torch.tensor([0, 1, 1])
})
...
```

## Importing one search space variant
```python3
from transformer_plus_plus.search_space.modeling_gpt2 import GPT2Config
from transformer_plus_plus.search_space.search_space import build_single_op_ss

ss = build_single_op_ss(d_inners=(768*3, 768*4), min_layers=1, max_layers=12,
                       attn_window_props=(0.25, 0.5, 0.75, 1.0),
                       hf_config=GPT2Config(n_positions=1024),
                       seed=1)
m = ss.random_sample()

print(m.archid)

# Torch module is in `m.arch`
m.arch.forward(**{
    'input_ids': torch.tensor([[1, 2, 3]]),
    'attention_mask': torch.tensor([0, 1, 1])
})

# Other ss operations
m2 = ss.mutate(m)
print(m2.archid)

m3 = ss.crossover([m, m2])
print(m3.archid)
```

## Running an experiment

```python3
from transformer_plus_plus.training.experiment import Experiment

exp = Experiment(
    arch_config='confs/gpt2_base/gpt2_base.json', 
    experiment_config='confs/gpt2_base/trainer_config.yaml',
    output_dir="/tmp/out"
)

exp.run()
```
