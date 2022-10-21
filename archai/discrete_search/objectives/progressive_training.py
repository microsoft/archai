from typing import Callable, List, Dict, Optional, Union
import tempfile

import ray
from overrides import overrides

from archai.discrete_search import (
    ArchaiModel, DiscreteSearchSpace,
    DatasetProvider, Objective, AsyncObjective
)


def ray_wrap_training_fn(training_fn):
    def stateful_training_fn(arch, dataset, budget, training_state: Optional[Dict] = None):
        metric_result, training_state = training_fn(arch, dataset, budget, training_state)
        return arch, metric_result, training_state
    
    return stateful_training_fn


class ProgressiveTraining(Objective):
    def __init__(self, search_space: DiscreteSearchSpace, 
                 training_fn: Callable, higher_is_better: bool = False):
        self.search_space = search_space
        self.training_fn = training_fn
        self.higher_is_better = higher_is_better

        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def evaluate(self, nas_model: ArchaiModel, dataset: DatasetProvider,
                budget: Optional[float] = None) -> float:
        # Tries to retrieve previous training state
        tr_state = self.training_states.get(nas_model.archid, None)

        # Computes metric and updates training state
        metric_result, updated_tr_state = self.training_fn(nas_model, dataset, budget, tr_state)
        self.training_states[nas_model.archid] = updated_tr_state

        return metric_result


class RayProgressiveTraining(AsyncObjective):
    def __init__(self, search_space: DiscreteSearchSpace, 
                 training_fn: Callable, higher_is_better: bool = False,
                 timeout: Optional[float] = None, force_stop: bool = False, 
                 **ray_kwargs):

        self.search_space = search_space
        
        if ray_kwargs:
            self.compute_fn = ray.remote(**ray_kwargs)(ray_wrap_training_fn(training_fn))
        else:
            self.compute_fn = ray.remote(ray_wrap_training_fn(training_fn))
        
        self.higher_is_better = higher_is_better
        self.timeout = timeout
        self.force_stop = force_stop

        # Buffer that stores original model references from `send` calls
        # to update weights after training is complete
        self.models = []

        # Ray training job object refs
        self.results_ref = []
        
        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def send(self, nas_model: ArchaiModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        # Stores original model reference
        self.models.append(nas_model)

        current_tr_state = self.training_states.get(nas_model.archid, None)
        self.results_ref.append(self.compute_fn.remote(
            nas_model, dataset, budget, current_tr_state
        ))

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None] * len(self.results_ref)

        # Fetchs training job results
        if not self.timeout:
            results = ray.get(self.results_ref, timeout=self.timeout)
        else:
            # Maps each object from the object_refs list to its index
            ref2idx = {ref: i for i, ref in enumerate(self.results_ref)}
            
            # Gets all results available within `self.timeout` seconds.
            complete_objs, incomplete_objs = ray.wait(
                self.results_ref, timeout=self.timeout,
                num_returns=len(self.results_ref)
            )
            partial_results = ray.get(complete_objs)
            
            for ref, result in zip(complete_objs, partial_results):
                results[ref2idx[ref]] = result

            for incomplete_obj in incomplete_objs:
                ray.cancel(incomplete_obj, force=self.force_stop)

        # Gathers metrics and syncs local references
        metric_results = []

        for job_id, job_results in enumerate(results):
            if job_results:
                trained_model, job_metric, training_state = job_results
                
                # Syncs model weights
                with tempfile.NamedTemporaryFile() as tmp:
                    self.search_space.save_model_weights(trained_model, tmp.name)
                    self.search_space.load_model_weights(self.models[job_id], tmp.name)
                
                # Syncs training state
                self.training_states[trained_model.archid] = training_state

                metric_results.append(job_metric)

        # Resets model and job buffers
        self.models = []
        self.results_ref = []

        return metric_results

