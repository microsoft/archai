import pytest
from overrides import overrides
from typing import List
from random import Random
import json
import os
import numpy as np

from archai.discrete_search import ArchaiModel, EvolutionarySearchSpace, BayesOptSearchSpace, SearchObjectives
from archai.discrete_search.api.predictor import Predictor, MeanVar
from archai.discrete_search.algos.bananas import MoBananasSearch
from archai.discrete_search.objectives.functional import EvaluationFunction


class DummySearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
        def __init__(self, seed: int = 10) -> None:
            self.rng = Random(seed)
        
        @overrides
        def random_sample(self) -> ArchaiModel:
            return ArchaiModel(None, archid=str(self.rng.randint(0, 100_000)))

        @overrides
        def mutate(self, arch: ArchaiModel) -> ArchaiModel:
            archid = arch.archid
            return ArchaiModel(None, str(int(archid) + self.rng.randint(-10, 10)))

        @overrides
        def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
            m1, m2 = arch_list[:2]
            new_archid = int((int(m1.archid) + int(m2.archid)) / 2)
            return ArchaiModel(None, str(new_archid))

        @overrides
        def save_arch(self, model: ArchaiModel, path: str) -> None:
            json.dump({'archid': model.archid}, open(path, 'w'))
        
        @overrides
        def load_arch(self, path: str) -> ArchaiModel:
            return ArchaiModel(None, json.load(path)['archid'])
        
        @overrides
        def load_model_weights(self, model: ArchaiModel, path: str) -> None:
            pass

        @overrides
        def save_model_weights(self, model: ArchaiModel, path: str) -> None:
            pass
        
        @overrides
        def encode(self, arch: ArchaiModel) -> np.ndarray:
            return np.array([int(arch.archid)])

@pytest.fixture
def search_objectives():
    rng1 = Random(1)
    rng2 = Random(2)
    rng3 = Random(3)

    o1 = EvaluationFunction(lambda m, d, b: rng1.random(), False)
    o2 = EvaluationFunction(lambda m, d, b: rng2.random(), False)
    r = EvaluationFunction(lambda m, d, b: rng3.random(), False)

    so = SearchObjectives()
    so.add_cheap_objective('Random1', o1, higher_is_better=False, constraint=(0.0, 0.4))
    so.add_expensive_objective('Random2', o2, higher_is_better=True)

    so.add_extra_constraint('Random3 constraint', r, constraint=(0.0, 0.6))

    return so


@pytest.fixture
def search_space():
    return DummySearchSpace()


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('out_bananas')


@pytest.fixture
def surrogate_model(search_objectives):
    class DummyPredictor(Predictor):
        def __init__(self, n_objs: int, seed1: int = 10, seed2: int = 20) -> None:
            self.n_objs = n_objs
            self.mean_rng = np.random.RandomState(seed1)
            self.var_rng = np.random.RandomState(seed2)

        @overrides
        def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
            pass

        @overrides
        def predict(self, encoded_archs: np.ndarray) -> MeanVar:
            n = len(encoded_archs)

            return MeanVar(
                self.mean_rng.random(size=(n, self.n_objs)),
                self.var_rng.random(size=(n, self.n_objs))
            )

    return DummyPredictor(len(search_objectives.exp_objs))


def test_bananas(output_dir, search_space, search_objectives, surrogate_model):
    algo = MoBananasSearch(
        output_dir, 
        search_space, search_objectives, dataset_provider=None, 
        surrogate_model=surrogate_model, num_iters=30, init_num_models=20,
        mutations_per_parent=10, num_parents=10, num_mutations=15
    )
    
    search_results = algo.search()
    assert len(os.listdir(output_dir)) > 0
    
    df = search_results.get_search_state_df()
    assert all(0 <= x <= 0.4 for x in df['Random1'].tolist())
    
    all_models = [m for iter_r in search_results.results for m in iter_r['models']]

    # Checks if all registered models satisfy constraints
    _, valid_models = search_objectives.eval_constraints(all_models, None)
    assert len(valid_models) == len(all_models)
