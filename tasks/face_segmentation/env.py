# test the conda environment.
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input, Output, dsl
import archai.common.azureml_helper as aml_helper

subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group_name = os.getenv('AML_RESOURCE_GROUP')
workspace_name = os.getenv('AML_WORKSPACE_NAME')

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)

cpu_compute_name = 'nas-cpu-cluster-D14-v2'

archai_job_env = aml_helper.create_environment_from_file(
    ml_client,
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    conda_file="conda.yaml",
    version='1.0.14')
environment_name = f"{archai_job_env.name}:{archai_job_env.version}"

test_component = command(experiment_name="testenv",
                         display_name="test environment",
                         compute=cpu_compute_name,
                         environment=environment_name,
                         code="test",
                         command="python test_env.py")


test_job = ml_client.create_or_update(test_component)
job_name = test_job.name
print(f'Started job: {job_name}')
