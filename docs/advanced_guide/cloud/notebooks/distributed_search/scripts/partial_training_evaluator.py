from typing import List, Optional, Union
from overrides import overrides
from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
from azure.ai.ml import MLClient, command
from azure.ai.ml import Input, Output

class PartialTrainingValAccuracy(AsyncModelEvaluator):
    def __init__(self, compute_cluster_name, environment_name, scripts_dir, training_epochs: float = 1.0):
        self.training_epochs = training_epochs
        self.compute_cluster_name = compute_cluster_name,
        self.environment_name = environment_name
        self.scripts_dir = scripts_dir
        self.jobs = []

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        if type(dataset) is not MnistDatasetProvider:
            raise Exception('Dataset must be of type MnistDatasetProvider')

        arch.arch.save_config('config.json')
        # create an AML job with 'train.py' to train this model.
        training_job = command(
            name="training " + arch.archid,
            display_name="Partial training of a NAS model",
            inputs={
                "data": Input(type="uri_folder")
            },
            outputs= {
                "quant_data": Output(type="uri_folder", mode="rw_mount")
            },

            # The source folder of the component
            code=self.scripts_dir,
            command="""python3 train.py \
                    --model_params ${{inputs.data}} \
                    --output ${{outputs.quant_data}} \
                    """,
            environment=self.environment_name,
        )


    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        pass