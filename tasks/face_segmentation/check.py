import os
from archai.discrete_search.search_spaces.config import ArchConfig
from archai.discrete_search.evaluators import TorchNumParameters
from archai.discrete_search.api.archai_model import ArchaiModel
from search_space.hgnet import StackedHourglass, HgnetSegmentationSearchSpace
from archai.common.config import Config


constraint = (1e5, 5e7)
evaluator = TorchNumParameters()
search_config = Config('confs/cpu_search.yaml')['search']
ss_config = search_config['search_space']
search_space = HgnetSegmentationSearchSpace(seed=1680312796, **ss_config.get('params', {}))
targets = os.path.join('archs', 'snp_target')

for file in os.listdir(targets):
    path = os.path.join(targets, file)
    if os.path.isfile(path) and path.endswith(".json"):
        config = ArchConfig.from_file(path)
        model = StackedHourglass(config, **search_space.model_kwargs)
        archid = os.path.splitext(file)[0]
        m = ArchaiModel(model, archid, config)
        num_params = evaluator.evaluate(m, None)
        if num_params < constraint[0] or num_params > constraint[1]:
            print(f"Model {file} has {num_params} parameters and is outside the valid range.")
