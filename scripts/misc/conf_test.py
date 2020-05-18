
from archai.common.common import common_init, expdir_abspath

def get_filepath(suffix):
    conf = common_init(config_filepath='confs/algos/darts.yaml',
        param_args=['--common.experiment_name', 'test_basename' + f'_{suffix}'
                    ])
    return expdir_abspath('somefile.txt')

print(get_filepath('search'))
print(get_filepath('eval'))
