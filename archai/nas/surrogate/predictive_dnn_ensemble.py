from typing import Tuple
from overrides.overrides import overrides

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from archai.common.common import logger
from archai.nas.surrogate.predictive_function import MeanVar, PredictiveFunction


    def __init__(self, num_features: int, num_layers: int = 5, width: int = 64,
                 num_ensemble_members: int = 5, sigmoid: bool = False):
        self.num_ensemble_members = num_ensemble_members
        self.num_layers = num_layers
        self.width = width
        self.lr = lr
        self.num_tr_steps = num_tr_steps

        # TODO: should have an architecture featurizer
        # object here and the featurizer should tell 
        # us what is the feature size
        # TODO: get from config
        self.input_feat_len = num_features
        self.num_layers = num_layers
        self.width = width
        self.sigmoid = sigmoid

        # build the ensemble
        self.ensemble = [FFEnsembleMember(input_feat_len=self.input_feat_len, 
                                          num_layers=self.num_layers, 
                                          width=self.width, sigmoid=self.sigmoid) 
                                          for _ in range(self.num_ensemble_members)]

        self.is_fit = False
        self.device = 'cuda'
        self.X_meanvar = None
        self.y_meanvar = None

    def to_cuda(self):
        for m in self.ensemble:
            m.cuda()
        self.device = 'cuda'

    def to_cpu(self):
        for m in self.ensemble:
            m.cpu()
        self.device = 'cpu'
    
    @overrides
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X.shape) == 2
        assert len(y.shape) == 2

        _, num_features = X.shape
        _, num_objectives = y.shape

        self.X_meansd = np.mean(X, axis=0), np.std(X, axis=0)
        self.y_meansd = np.mean(y, axis=0), np.std(y, axis=0)

        # Init ensemble models
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
        for member in tqdm(self.ensemble, desc='Training DNN Ensemble...'):
            criterion = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(member.parameters(), lr=self.lr)
            member.train()

            for t in range(self.num_tr_steps):
                y_pred = member(Xt)
                loss = criterion(y_pred.squeeze(), yt.squeeze())        

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'Final training loss {loss.item()}')

        self.is_fit = True

    @overrides
    def predict(self, X: np.ndarray) -> MeanVar:
        assert len(X.shape) == 2
        assert self.is_fit, 'PredictiveDNNEnsemble: predict called before fit!' 

        X = (X.copy() - self.X_meansd[0]) / (self.X_meansd[1] + 1e-7)
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)

        preds = []
        with torch.no_grad():
            for member in self.ensemble:
                member.eval()
                pred = member(Xt).to('cpu').numpy()
                preds.append(pred * (self.y_meansd[1] + 1e-7) + self.y_meansd[0])
        
        preds = np.array(preds)
        return MeanVar(mean=np.mean(preds, axis=0), var=np.var(preds, axis=0))


class FFEnsembleMember(nn.Module):
    def __init__(self, input_feat_len:int=128, num_layers:int=10, width:int=20, sigmoid: bool = False):
        super(FFEnsembleMember, self).__init__()

        self.input_feat_len = input_feat_len
        self.num_layers = num_layers
        self.width = width

        self.linears = nn.ModuleList([nn.Linear(self.input_feat_len, width)])
        self.linears.extend([nn.Linear(width, width) for i in range(1, self.num_layers-1)])
        
        output_layers = [
            nn.Linear(width, 1)
        ]
        
        if sigmoid: 
            output_layers.append(nn.Sigmoid())
        
        self.output = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears:
            x = f.relu(layer(x))
        return self.output(x)
