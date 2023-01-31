# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from archai.discrete_search.algos.regularized_evolution import (
    RegularizedEvolutionSearch,
)


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("out_re")


def test_regularized_evolution(output_dir, search_space, search_objectives, surrogate_model):
    algo = RegularizedEvolutionSearch(
        search_space=search_space,
        search_objectives=search_objectives,
        dataset_provider=None,
        output_dir=output_dir,
        num_iters=5,
        init_num_models=4,
        pareto_sample_size=4,
        history_size=10,
        seed=1,
    )

    search_results = algo.search()
    assert len(os.listdir(output_dir)) > 0

    df = search_results.get_search_state_df()
    assert all(0 <= x <= 0.4 for x in df["Random1"].tolist())

    all_models = [m for iter_r in search_results.results for m in iter_r["models"]]

    # Checks if all registered models satisfy constraints
    _, valid_models = search_objectives.validate_constraints(all_models, None)
    assert len(valid_models) == len(all_models)
