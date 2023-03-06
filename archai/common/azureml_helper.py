from typing import Dict, Union, Optional, Any
from pathlib import Path

from azure.ai.ml import MLClient, command, load_job
from azure.ai.ml.entities import AmlCompute, Environment, Command, Job
from azure.identity import DefaultAzureCredential

def get_aml_client_from_file(config_path: Union[str, Path]) -> MLClient:
    """ Creates an MLClient object from a workspace config file

    Args:
        config_path (Union[str, Path]): Path to the workspace config file

    Returns:
        MLClient: MLClient object
    """
    credential = DefaultAzureCredential()
    config_path = Path(config_path)

    ml_client = MLClient.from_config(
        credential=credential,
        path=config_path
    )
    return ml_client


def create_compute_cluster(
        ml_client: MLClient,
        compute_name: str,
        type: Optional[str] = "amlcompute",
        size: Optional[str] = "Standard_D14_v2",
        min_instances: Optional[int] = 0,
        max_instances: Optional[int] = 4,
        idle_time_before_scale_down: Optional[int] = 180,
        tier: Optional[str] = "Dedicated",
        **kwargs):
    """ Creates a compute cluster for the workspace

    Args:
        ml_client (MLClient): MLClient object
        compute_name (str): Name of the (CPU/GPU) compute cluster
        type (str, optional): Type of the compute cluster. Defaults to "amlcompute".
        size (str, optional): VM Family of the compute cluster. Defaults to "Standard_D14_v2".
        min_instances (int, optional): Minimum running nodes when there is no job running. Defaults to 0.
        max_instances (int, optional): Maximum number of nodes in the cluster. Defaults to 4.
        idle_time_before_scale_down (int, optional): How many seconds will the node running after the job termination. Defaults to 180.
        tier (str, optional): Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination. Defaults to "Dedicated".

    Returns:
        None
    """
    try:
        cpu_cluster = ml_client.compute.get(compute_name)
        print(f"You already have a cluster named {compute_name}, we'll reuse it as is.")
    except Exception:
        cpu_compute = AmlCompute(
            name=compute_name,
            type=type,
            size=size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time_before_scale_down,
            tier=tier,
            **kwargs
        )

        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_compute).result()
        print(f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}")


def create_environment_from_file(
        ml_client: MLClient,
        custom_env_name: Optional[str] = "aml-archai",
        description: Optional[str] = "Custom environment for Archai",
        tags: Optional[Dict[str, Any]] = None,
        conda_file: Optional[str] = "conda.yaml",
        image: Optional[str] = "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        version: Optional[str] = "0.1.0",
        **kwargs) -> Environment:
    """ Creates an environment from a conda file

    Args:
        ml_client (MLClient): MLClient object
        custom_env_name (str, optional): Name of the environment. Defaults to "aml-archai".
        description (str, optional): Description of the environment. Defaults to "Custom environment for Archai".
        tags (Dict[str, Any], optional): Tags for the environment, e.g. {"archai": "1.0.0"}. Defaults to None.
        conda_file (str, optional): Path to the conda file. Defaults to "conda.yaml".
        image (str, optional): Docker image for the environment. Defaults to "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest".
        version (str, optional): Version of the environment. Defaults to "0.1.0".

    Returns:
        Environment: Environment object
    """

    tags = tags or {"archai": "1.0.0"}

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

# TODO remove
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

# TODO remove
def create_job_from_file(source: Union[str, Path], **kwargs) -> Job:
    return load_job(source=source, **kwargs)

# TODO remove
def run_job(ml_client: MLClient, job: Job) -> Job:
    return ml_client.create_or_update(job)

# TODO remove
def stream_job_logs(ml_client: MLClient, job: Job):
    ml_client.jobs.stream(job.name)


# TODO How can we return the path that the output was downloaded to?
def download_job_output(
        ml_client: MLClient,
        job_name: str,
        output_name: str,
        download_path: Optional[Union[str, Path]] = "output") -> None:
    """ Downloads the output of a job

    Args:
        ml_client (MLClient): MLClient object
        job_name (str): Name of the job
        output_name (str): Named output to downlaod
        download_path (Union[str, Path], optional): Path to download the output to. Defaults to "output".

    Returns:
        None
    """
    try:
        target_job = ml_client.jobs.get(job_name)
    except Exception as e:
        print(f"{e.error}")
        return None

    if target_job.status == "Completed":
        ml_client.jobs.download(name=target_job.name, download_path=Path(download_path), output_name=output_name)
    else:
        print(f"Job {target_job.name} is not completed yet")
