import torch
import torch
import time
from tqdm import tqdm

from archai.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.metrics.torch_model import FlopsMetric, NumParametersMetric
from archai.metrics.functional import FunctionalMetric
from archai.metrics.ray import RayParallelMetric
from archai.metrics import evaluate_models, get_pareto_frontier

from archai.algos.bananas.bananas_search import MoBananasSearch
from archai.search_spaces.discrete.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.metrics.lookup import NatsBenchMetric

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider

import torch
import time
from tqdm import tqdm

from archai.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.metrics.torch_model import FlopsMetric, NumParametersMetric
from archai.metrics.functional import FunctionalMetric
from archai.metrics.ray import RayParallelMetric
from archai.metrics.progressive_training import RayProgressiveTrainingMetric
from archai.metrics import evaluate_models, get_pareto_frontier

from archai.algos.evolution_pareto.evolution_pareto_search import EvolutionParetoSearch
from archai.algos.sucessive_halving.sucessive_halving import SucessiveHalvingAlgo
from archai.datasets.providers.lmdb_image_provider import TensorpackLmdbImageProvider
from archai.search_spaces.discrete.segmentation_dag.search_space import SegmentationDagSearchSpace

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider

## SegmentationDag Search space
from archai.search_spaces.discrete.segmentation_dag.search_space import SegmentationDagSearchSpace
from archai.datasets.providers.lmdb_image_provider import TensorpackLmdbImageProvider

ss = SegmentationDagSearchSpace(
    nb_classes=1, img_size=(160, 96), min_layers=3, max_layers=5, min_mac=1e5,
    op_subset='conv3x3,conv5x5'
)

dataset = TensorpackLmdbImageProvider({
    'dataroot': '/home/pkauffmann/dataroot/',
    'tr_lmdb': 'segdata/cached_asg_data_portrait/epoch_0.lmdb',
    'te_lmdb': 'segdata/cached_asg_data_portrait/validation.lmdb',
    'img_key': 'img', 'mask_key': 'seg', 'is_bgr': False
})

def partial_training_val_iou(model, dataset_provider, budget: float = 1.0):
    tr_dataset, val_dataset = dataset_provider.get_datasets(True, True, None, None)
    tr_dl = torch.utils.data.DataLoader(tr_dataset, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    arch = model.arch.to('cuda')
    
    opt = torch.optim.Adam(arch.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    if budget:
        tr_steps = 50 * budget
    else:
        tr_steps = 50
    
    print(f'Training for {tr_steps}')

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

    arch.to('cpu')
    return (val_loss / (i + 1)).item()


# Objective list
objectives = {
    'ONNX Latency': RayParallelMetric(AvgOnnxLatencyMetric(input_shape=(1, 3, 96, 160)), num_cpus=1.0),
    'Test accuracy': FunctionalMetric(partial_training_val_iou, higher_is_better=True)
}

# Search
algo = SucessiveHalvingAlgo(
    ss, objectives, dataset_provider=dataset, num_iters=4,
    init_num_models=8, init_budget=1.0,
    output_dir='/home/pkauffmann/logdir/sucessive_halving3_natsbench'
)
algo.search()
