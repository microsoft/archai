# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nas.model_desc import ModelDesc
from archai.common.common import common_init
from archai.nas.model import Model
from archai.petridish.petridish_model_desc_builder import PetridishModelBuilder
from archai.nas.nas_utils import create_macro_desc

from archai.common.model_summary import summary

conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                    param_args=['--common.experiment_name', 'petridish_run2_seed42_eval'])

conf_eval = conf['nas']['eval']
conf_model_desc   = conf_eval['model_desc']

conf_model_desc['n_cells'] = 14
template_model_desc = ModelDesc.load('$expdir/final_model_desc.yaml')
model_desc = create_macro_desc(conf_model_desc, template_model_desc)

mb = PetridishModelBuilder()
mb.register_ops()
model = Model(model_desc, droppath=False, affine=False)

summary(model, [64, 3, 32, 32])


exit(0)