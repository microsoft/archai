Quick Start
===========

In this quickstart example, we will apply Archai in Natural Language Processing to find the optimal Pareto-frontier Transformers' configurations according to a set of objectives.

Creating the Search Space
-------------------------

We start by importing the `TransformerFlexSearchSpace` class which represents the search space for the Transformer architecture:

.. code-block:: python

    from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import TransformerFlexSearchSpace

    space = TransformerFlexSearchSpace("gpt2")

Defining Search Objectives
--------------------------

Next, we define the objectives we want to optimize. In this example, we use `NonEmbeddingParamsProxy`, `TransformerFlexOnnxLatency`, and `TransformerFlexOnnxMemory` to define the objectives:

.. code-block:: python

    from archai.discrete_search.api.search_objectives import SearchObjectives
    from archai.discrete_search.evaluators.nlp.parameters import NonEmbeddingParamsProxy
    from archai.discrete_search.evaluators.nlp.transformer_flex_latency import TransformerFlexOnnxLatency
    from archai.discrete_search.evaluators.nlp.transformer_flex_memory import TransformerFlexOnnxMemory

    search_objectives = SearchObjectives()
    search_objectives.add_objective(
        "non_embedding_params",
        NonEmbeddingParamsProxy(),
        higher_is_better=True,
        compute_intensive=False,
        constraint=(1e6, 1e9),
    )
    search_objectives.add_objective(
        "onnx_latency",
        TransformerFlexOnnxLatency(space),
        higher_is_better=False,
        compute_intensive=False,
    )
    search_objectives.add_objective(
        "onnx_memory",
        TransformerFlexOnnxMemory(space),
        higher_is_better=False,
        compute_intensive=False,
    )

Initializing the Algorithm
--------------------------

We use the `EvolutionParetoSearch` algorithm to conduct the search:

.. code-block:: python

    from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch

    algo = EvolutionParetoSearch(
        space,
        search_objectives,
        None,
        "tmp",
        num_iters=5,
        init_num_models=10,
        seed=1234,
    )

Performing the Search
---------------------

Finally, we call the `search()` method to start the NAS process:

.. code-block:: python

    algo.search()

The algorithm will iterate through different network architectures, evaluate their performance based on the defined objectives, and ultimately produce a frontier of Pareto-optimal results.
