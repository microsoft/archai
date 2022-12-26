# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorwatch as tw
from archai.cv import models

model_names = ['resnet18', 'resnet34', 'resnet101', 'densenet121']

for model_name in model_names:
    model = getattr(models, model_name)()
    model_stats = tw.ModelStats(model,  [1, 3, 224, 224], clone_model=False)
    print(f'{model_name}: flops={model_stats.Flops}, parameters={model_stats.parameters}, memory={model_stats.inference_memory}')