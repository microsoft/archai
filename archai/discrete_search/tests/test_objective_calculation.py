import torch
import time
import random
from tqdm import tqdm

from archai.discrete_search import NasModel, evaluate_models, SearchResults, DatasetProvider
from archai.discrete_search.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.discrete_search.metrics.torch_model import FlopsMetric, NumParametersMetric
from archai.discrete_search.metrics.functional import FunctionalMetric
from archai.discrete_search.metrics.ray import RayParallelMetric

from archai.discrete_search.datasets.lmdb_image_provider import TensorpackLmdbImageProvider
from archai.discrete_search.search_spaces.segmentation_dag.search_space import SegmentationDagSearchSpace

from archai.datasets.dataset_provider import DatasetProvider

dataset_provider = TensorpackLmdbImageProvider({
    'dataroot': '/home/pkauffmann/dataroot/',
    'tr_lmdb': 'segdata/cached_asg_data_portrait/epoch_0.lmdb',
    'te_lmdb': 'segdata/cached_asg_data_portrait/validation.lmdb',
    'is_bgr': False, 'img_key': 'img', 'mask_key': 'seg'
})

# Segmentation search space
ss = SegmentationDagSearchSpace(nb_classes=1, img_size=(160, 96), min_layers=3, max_layers=5)

# Gets 10 models
model_list = [ss.random_sample() for i in range(10)]


def validation_loss(model: NasModel, dataset_provider: DatasetProvider, budget = None):
    tr_dataset, val_dataset = dataset_provider.get_train_val_datasets()
    tr_dl = torch.utils.data.DataLoader(tr_dataset, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    arch = model.arch.to('cuda')
    
    opt = torch.optim.Adam(arch.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    tr_steps = 50

    arch.train()
    for i, batch in enumerate(tqdm(tr_dl, total=tr_steps)):
        if i == tr_steps:
            break

        opt.zero_grad()
        img, mask = batch['image'], batch['mask']

        pred = arch(img.to('cuda'))
        loss = criterion(pred.squeeze(), mask.to('cuda') / 255.0)

        loss.backward()
        opt.step()

    arch.eval()
    with torch.no_grad():
        val_loss = 0.0

        for i, batch in enumerate(tqdm(val_dl, desc='validating...')):
            if i == tr_steps:
                break

            img, mask = batch['image'], batch['mask']

            pred = arch(img.to('cuda'))
            val_loss += criterion(pred.squeeze(), mask.to('cuda') / 255.0)

    return (val_loss / (i + 1)).item()


# Objective dict
objectives = {
    # Synchronous metrics:
    'Number of parameters': NumParametersMetric(input_shape=(1, 3, 160, 96)),
    #AvgOnnxLatencyMetric(input_shape=(1, 3, 160, 96)),

    # RayParallelMetric wraps a regular metric as an async metric that performs the calculations in parallel using Ray
    'Validation Loss': RayParallelMetric(
        FunctionalMetric(evaluation_fn=validation_loss, higher_is_better=False),
        timeout=900, num_gpus=0.2, max_calls=1
    )
}

# Results
search_state = SearchResults(search_space=ss, objectives=objectives)
results = evaluate_models(model_list, objectives, dataset_provider)

search_state.add_iteration_results(model_list, results, extra_model_data={
    'archid': [m.archid for m in model_list]
})

#search_state.save_2d_pareto_evolution_plot((1, 2), '/tmp/plot.png')

# Next iter
model_list = [ss.random_sample() for i in range(10, 15)]

results = evaluate_models(model_list, objectives, dataset_provider)
search_state.add_iteration_results(model_list, results, extra_model_data={
    'archid': [m.archid for m in model_list]
})
search_state.save_all_2d_pareto_evolution_plots('/home/pkauffmann/logdir/sample_out/')
#search_state.save_2d_pareto_evolution_plot((1, 2), '/tmp/plot.png')

# Next iter
model_list = [ss.random_sample() for i in range(10, 15)]

results = evaluate_models(model_list, objectives, dataset_provider)
search_state.add_iteration_results(model_list, results, extra_model_data={
    'archid': [m.archid for m in model_list]
})

model_list = [ss.random_sample() for i in range(10, 15)]

results = evaluate_models(model_list, objectives, dataset_provider)
search_state.add_iteration_results(model_list, results, extra_model_data={
    'archid': [m.archid for m in model_list]
})


model_list = [ss.random_sample() for i in range(10, 15)]

results = evaluate_models(model_list, objectives, dataset_provider)
search_state.add_iteration_results(model_list, results, extra_model_data={
    'archid': [m.archid for m in model_list]
})


search_state.save_all_2d_pareto_evolution_plots('/home/pkauffmann/logdir/sample_out/')
