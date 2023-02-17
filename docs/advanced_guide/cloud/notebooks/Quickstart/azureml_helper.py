from typing import Dict, Union
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, Environment, Command, Job
from azure.identity import DefaultAzureCredential

# TODO Move this module to Archai utils stuff

def get_aml_client_from_file(config_path: Union[str, Path] = "../.azureml/config.json") -> MLClient:
    credential = DefaultAzureCredential()
    config_path = Path(config_path)

    ml_client = MLClient.from_config(
        credential=credential,
        path=config_path
    )
    return ml_client


def create_compute_cluster(
        ml_client: MLClient,
        cpu_compute_name: str,
        type: str = "amlcompute",
        size: str = "Standard_D14_v2",
        min_instances: int = 0,
        max_instances: int = 4,
        idle_time_before_scale_down: int = 180,
        tier: str = "Dedicated",
        **kwargs):
    try:
        cpu_cluster = ml_client.compute.get(cpu_compute_name)
        print(f"You already have a cluster named {cpu_compute_name}, we'll reuse it as is.")
    except Exception:
        cpu_compute = AmlCompute(
            name=cpu_compute_name,
            type=type,
            # VM Family
            size=size,
            # Minimum running nodes when there is no job running
            min_instances=min_instances,
            # Nodes in cluster
            max_instances=max_instances,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=idle_time_before_scale_down,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier=tier,
            **kwargs
        )

        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_compute).result()
        print(f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}")


def create_environment_from_file(
        ml_client: MLClient,
        custom_env_name: str = "aml-archai",
        description: str = "Custom environment for Archai",
        tags: dict = {"archai": "1.0.0"},
        conda_file: str = "conda.yaml",
        image: str = "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        version: str = "0.1.0",
        **kwargs) -> Environment:

    archai_job_env = Environment(
        name=custom_env_name,
        description=description,
        tags=tags,
        conda_file=conda_file,
        image=image,
        version=version,
        **kwargs
    )
    archai_job_env = ml_client.environments.create_or_update(archai_job_env)

    print(
        f"Environment with name {archai_job_env.name} is registered to workspace, the environment version is {archai_job_env.version}")

    return archai_job_env


def create_job(
        experiment_name: str,
        display_name: str,
        compute_name: str,
        environment: str,
        code: str,
        cli_command: str,
        outputs: Dict,
        **kwargs) -> Command:

    job = command(
        display_name=display_name,
        outputs=outputs,
        code=code,
        command=cli_command,
        environment=environment,
        compute=compute_name,
        experiment_name=experiment_name,
        **kwargs
    )

    return job


def run_job(ml_client: MLClient, job: Job) -> Job:
    return ml_client.create_or_update(job)


def stream_job_logs(ml_client: MLClient, job: Job):
    ml_client.jobs.stream(job.name)


# TODO How can we return the path that the output was downloaded to?
def download_job_output(
        ml_client: MLClient,
        job_name: str,
        output_name: str,
        download_path: Union[str, Path] = "output") -> None:

    try:
        target_job = ml_client.jobs.get(job_name)
    except Exception as e:
        print(f"{e.error}")
        return None

    if target_job.status == "Completed":
        ml_client.jobs.download(name=target_job.name, download_path=Path(download_path), output_name=output_name)
    else:
        print(f"Job {target_job.name} is not completed yet")
