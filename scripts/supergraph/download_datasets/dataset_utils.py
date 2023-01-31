from archai.common.config import Config


def get_dataroot() -> str:
    conf = Config(file_path="confs/algos/manual.yaml")
    return conf["dataset"]["dataroot"]
