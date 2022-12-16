import pytest
from overrides import overrides
from typing import List
from random import Random
import json
import os

from archai.discrete_search import ArchaiModel, EvolutionarySearchSpace, SearchObjectives
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.objectives.functional import EvaluationFunction

class DummySearchSpace(EvolutionarySearchSpace):
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
    return tmp_path_factory.mktemp('out_evo')


def test_pareto_evolution(output_dir, search_space, search_objectives):
    algo = EvolutionParetoSearch(
        search_space, search_objectives, None, output_dir, num_iters=30, 
        init_num_models=20
    )
    
    search_results = algo.search()
    assert len(os.listdir(output_dir)) > 0

    df = search_results.get_search_state_df()
    assert all(0 <= x <= 0.4 for x in df['Random1'].tolist())
    
    all_models = [m for iter_r in search_results.results for m in iter_r['models']]

    # Checks if all registered models satisfy constraints
    _, valid_models = search_objectives.eval_constraints(all_models, None)
    assert len(valid_models) == len(all_models)

