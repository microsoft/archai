# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.common.common import common_init
from archai.common.model_summary import summary
from archai.supergraph.algos.petridish.petridish_model_desc_builder import (
    PetridishModelBuilder,
)
from archai.supergraph.nas.model import Model
from archai.supergraph.nas.model_desc import ModelDesc

conf = common_init(
    config_filepath="confs/petridish_cifar.yaml", param_args=["--common.experiment_name", "petridish_run2_seed42_eval"]
)

conf_eval = conf["nas"]["eval"]
conf_model_desc = conf_eval["model_desc"]

conf_model_desc["n_cells"] = 14
template_model_desc = ModelDesc.load("$expdir/final_model_desc.yaml")

model_builder = PetridishModelBuilder()

model_desc = model_builder.build(conf_model_desc, template=template_model_desc)

mb = PetridishModelBuilder()
model = Model(model_desc, droppath=False, affine=False)

summary(model, [64, 3, 32, 32])


exit(0)
