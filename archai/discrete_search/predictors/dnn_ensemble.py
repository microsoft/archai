# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from overrides import overrides
from tqdm import tqdm

from archai.discrete_search.api.predictor import MeanVar, Predictor


class PredictiveDNNEnsemble(Predictor):
    """Deep Neural Network Ensemble predictor.

    Predicts the outcome of a set of expensive objectives using an ensemble of MLP models.

    """

    def __init__(
        self,
        num_ensemble_members: Optional[int] = 5,
        num_layers: Optional[int] = 5,
        width: Optional[int] = 64,
        lr: Optional[float] = 1e-4,
        num_tr_steps: Optional[int] = 2_000,
    ) -> None:
        """Initializes the predictor.

        Args:
            num_ensemble_members: Number of ensemble members.
            num_layers: Number of layers of each member.
            width: Number of neurons in each hidden layer.
            lr: Learning rate of each ensemble mmember.
            num_tr_steps: Number of training steps of each member.

        """

        self.num_ensemble_members = num_ensemble_members
        self.num_layers = num_layers
        self.width = width
        self.lr = lr
        self.num_tr_steps = num_tr_steps

        self.is_fit = False
        self.device = "cuda"
        self.X_meanvar = None
        self.y_meanvar = None

    def to_cuda(self) -> None:
        """Moves the predictor to CUDA."""

        for m in self.ensemble:
            m.cuda()
        self.device = "cuda"

    def to_cpu(self) -> None:
        """Moves the predictor to CPU."""

        for m in self.ensemble:
            m.cpu()
        self.device = "cpu"

    @overrides
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X.shape) == 2
        assert len(y.shape) == 2

        _, num_features = X.shape
        _, num_objectives = y.shape

        self.X_meansd = np.mean(X, axis=0), np.std(X, axis=0)
        self.y_meansd = np.mean(y, axis=0), np.std(y, axis=0)

        # Initialize ensemble models
        self.ensemble = [
            FFEnsembleMember(num_objectives, num_features, self.num_layers, self.width).to(self.device)
            for _ in range(self.num_ensemble_members)
        ]

        # Normalizes features and targets
        X = (X.copy() - self.X_meansd[0]) / (self.X_meansd[1] + 1e-7)
        y = (y.copy() - self.y_meansd[0]) / (self.y_meansd[1] + 1e-7)

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.device)

        # TODO: should we be splitting data into
        # train and val?
        for member in tqdm(self.ensemble, desc="Training DNN Ensemble..."):
            criterion = torch.nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(member.parameters(), lr=self.lr)
            member.train()

            for t in range(self.num_tr_steps):
                y_pred = member(Xt)
                loss = criterion(y_pred.squeeze(), yt.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.is_fit = True

    @overrides
    def predict(self, X: np.ndarray) -> MeanVar:
        assert len(X.shape) == 2
        assert self.is_fit, "PredictiveDNNEnsemble: predict called before fit!"

        X = (X.copy() - self.X_meansd[0]) / (self.X_meansd[1] + 1e-7)
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)

        preds = []
        with torch.no_grad():
            for member in self.ensemble:
                member.eval()
                pred = member(Xt).to("cpu").numpy()
                preds.append(pred * (self.y_meansd[1] + 1e-7) + self.y_meansd[0])

        preds = np.array(preds)

        return MeanVar(mean=np.mean(preds, axis=0), var=np.var(preds, axis=0))


class FFEnsembleMember(nn.Module):
    """Feedforward ensemble member."""

    def __init__(
        self,
        num_objectives: Optional[int] = 1,
        input_feat_len: Optional[int] = 128,
        num_layers: Optional[int] = 10,
        width: Optional[int] = 20,
    ) -> None:
        """Initializes the ensemble member.

        Args:
            num_objectives: Number of objectives.
            input_feat_len: Length of input features.
            num_layers: Number of layers.
            width: Width of each layer.

        """

        super(FFEnsembleMember, self).__init__()

        self.input_feat_len = input_feat_len
        self.num_layers = num_layers
        self.width = width

        self.linears = nn.ModuleList([nn.Linear(self.input_feat_len, width)])
        self.linears.extend([nn.Linear(width, width) for i in range(1, self.num_layers - 1)])
        self.output = nn.Linear(width, num_objectives)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears:
            x = f.relu(layer(x))

        return self.output(x)
