# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from archai.discrete_search.algos.local_search import LocalSearch


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("out")


def test_local_search(output_dir, search_space, search_objectives):
    algo = LocalSearch(
        search_space,
        search_objectives,
        None,
        output_dir,
        num_iters=2,
        init_num_models=5,
        mutations_per_parent=2,
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
