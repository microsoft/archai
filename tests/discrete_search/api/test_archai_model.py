# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.discrete_search.api.archai_model import ArchaiModel


def test_archai_model():
    model = torch.nn.Linear(10, 1)
    archid = "test_archid"
    metadata = {"key": "value"}

    # Assert that attributes are set correctly
    archai_model = ArchaiModel(model, archid, metadata)
    assert archai_model.arch == model
    assert archai_model.archid == archid
    assert archai_model.metadata == metadata
    assert (
        str(archai_model)
        == "ArchaiModel(\n\tarchid=test_archid, \n\tmetadata={'key': 'value'}, \n\tarch=Linear(in_features=10, out_features=1, bias=True)\n)"
    )
