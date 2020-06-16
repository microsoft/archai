# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from archai.common.common import common_init
from archai.common import utils
import os

def get_filepath(suffix):
    conf = common_init(config_filepath='confs/algos/darts.yaml',
        param_args=['--common.experiment_name', 'test_basename' + f'_{suffix}'
                    ])
    return utils.full_path(os.path.join('$expdir' ,'somefile.txt'))


print(get_filepath('search'))
print(get_filepath('eval'))
