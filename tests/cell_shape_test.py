# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# import torch

# from archai.common.common import common_init
# from archai.darts.darts_cell_builder import DartsCellBuilder
# from archai.nas.nas_utils import create_model

# def test_cell_shape():
#     conf = common_init(config_filepath=None,
#                        param_args=['--common.logdir', '',
#                                    '--common.experiment_name', 'unit_test'])

#     conf_dataset = conf['dataset']
#     conf_search = conf['nas']['search']
#     conf_model_desc = conf_search['model_desc']

#     cell_builder = DartsCellBuilder()
#     device = torch.device('cuda')

#     model = create_model(conf_model_desc, device,
#                           affine=False, droppath=False,
#                          cell_builder=None, template_model_desc=None)

#     x = torch.randn(64, 3, 32, 32).to(device)

#     s0 = model._stem0_op(x)
#     s1 = model._stem1_op(x)

#     assert list(s0.shape) == [64, 48, 32, 32]

#     logits_aux = None
#     for cell in model._cells:
#         s0, s1 = s1, cell.forward(s0, s1)
#         if cell.aux_tower is not None:
#             logits_aux = cell.aux_tower(s1)

#     # s1 is now the last cell's output
#     out = model.pool_op(s1)
#     logits = model.linear(out.view(out.size(0), -1))  # flatten
