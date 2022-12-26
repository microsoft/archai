from archai.common.config import Config

def get_dataroot()->str:
    conf = Config(config_filepath='benchmarks/confs/algos/manual.yaml')
    return conf['dataset']['dataroot']

