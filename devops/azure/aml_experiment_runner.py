import sys

import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.core.container_registry import ContainerRegistry
from azureml.train.estimator import Estimator
from azureml.core import Environment

from archai.common.config import Config


class AmlExperimentRunner():
    def __init__(self, config_filepath:str) -> None:
        
        # read in config
        self.conf = Config(config_filepath)

        # config region
        self.conf_aml = self.conf['aml_config']
        self.conf_storage = self.conf['storage']
        self.conf_cluster = self.conf['cluster_config']
        self.conf_docker = self.conf['azure_docker']
        self.conf_experiment = self.conf['experiment']
        # end region

        # initialize workspace
        self.ws = Workspace.from_config(path=self.conf_aml['aml_config_file'])
        print('Workspace name: ' + self.ws.name,
          'Azure region: ' + self.ws.location,
          'Subscription id: ' + self.ws.subscription_id,
          'Resource group: ' + self.ws.resource_group, sep='\n')

        # register blobs
        # TODO: make blob registration more flexible
        self.input_ds = Datastore.register_azure_blob_container(workspace=self.ws,
                                                       datastore_name=self.conf_storage['input_datastore_name'],
                                                       container_name=self.conf_storage['input_container_name'],
                                                       account_name=self.conf_storage['input_azure_storage_account_name'],
                                                       account_key=self.conf_storage['input_azure_storage_account_key'],
                                                       create_if_not_exists=False)

        self.output_ds = Datastore.register_azure_blob_container(workspace=self.ws,
                                                       datastore_name=self.conf_storage['output_datastore_name'],
                                                       container_name=self.conf_storage['output_container_name'],
                                                       account_name=self.conf_storage['output_azure_storage_account_name'],
                                                       account_key=self.conf_storage['output_azure_storage_account_key'],
                                                       create_if_not_exists=False)

        # create compute cluster
        try:
            self.compute_target = ComputeTarget(workspace=self.ws, name=self.conf_cluster['cluster_name'])
            print(self.compute_target.get_status().serialize())
        except Exception as e:
            print('Encountered error trying to get the compute target')
            print(f'Exception was {e}')
            sys.exit(1)

        self.project_folder = self.conf_experiment['project_folder']

        # setup custom docker usage
        self.image_registry_details = ContainerRegistry()
        self.image_registry_details.address = self.conf_docker['image_registry_address']
        self.image_registry_details.username = self.conf_docker['image_registry_username']
        self.image_registry_details.password = self.conf_docker['image_registry_password']

        self.user_managed_dependencies = True

    # TODO: Make this into property?
    @property
    def input_datastore_handle(self):
        return self.input_ds

    # TODO: Make this into property?
    @property
    def output_datastore_handle(self):
        return self.output_ds

    def launch_experiment(self, exp_name:str, script_params:dict, entry_script:str):
        experiment = Experiment(self.ws, name=exp_name)
        
        est = Estimator(source_directory=self.project_folder,
                    script_params=script_params,
                    compute_target=self.compute_target,
                    entry_script=entry_script,
                    custom_docker_image=self.conf_docker['image_name'],
                    image_registry_details=self.image_registry_details,
                    user_managed=self.user_managed_dependencies,
                    source_directory_data_store=self.input_ds)

        run = experiment.submit(est)
        



        






    




