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
from archai.datasets.providers.lmdb_image_provider import TensorpackLmdbImageProvider
from archai.search_spaces.discrete.segmentation_dag.search_space import SegmentationDagSearchSpace

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider

dataset_provider = TensorpackLmdbImageProvider({
    'dataroot': '/home/pkauffmann/dataroot/',
    'tr_lmdb': 'segdata/cached_asg_data_portrait/epoch_0.lmdb',
    'te_lmdb': 'segdata/cached_asg_data_portrait/validation.lmdb',
    'is_bgr': False, 'img_key': 'img', 'mask_key': 'seg'
})

# Segmentation search space
ss = SegmentationDagSearchSpace(nb_classes=1, img_size=(160, 96), min_layers=6, max_layers=14)


def validation_loss(model: ArchWithMetaData, dataset_provider: DatasetProvider, budget = None):
    tr_dataset, val_dataset = dataset_provider.get_datasets(True, True, None, None)
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

# Objective list
objectives = {
    'ONNX Latency': RayParallelMetric(AvgOnnxLatencyMetric(input_shape=(1, 3, 64, 64)), num_cpus=1.0),
    #'Number of parameters': NumParametersMetric(input_shape=(1, 3, 64, 64)),
    'Test accuracy': NatsBenchMetric(ss, metric_name='test-accuracy', higher_is_better=True)
}

# Search
algo = MoBananasSearch(
    '/home/pkauffmann/logdir/bananas_natsbench',
    ss, objectives, dataset_provider=None,
    cheap_objectives=['ONNX Latency'],
    num_iters=10, num_parents=20, mutations_per_parent=2,
    num_mutations=20, init_num_models=10
)
algo.search()
