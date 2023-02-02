# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from archai.discrete_search.predictors.dnn_ensemble import PredictiveDNNEnsemble


def test_dnn_ensemble():
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100, 2)
    X_test = np.random.rand(50, 5)

    predictor = PredictiveDNNEnsemble(device="cpu")
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    assert y_pred.mean.shape == (50, 2)
    assert y_pred.var.shape == (50, 2)
