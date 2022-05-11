from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.predictive_function import MeanVar, PredictiveFunction




class PredictiveDNNEnsemble(PredictiveFunction):

    def __init__(self, num_ensemble_members:int=5):
        self.num_ensemble_members = num_ensemble_members

        # TODO: should have an architecture featurizer
        # object here and the featurizer should tell 
        # us what is the feature size
        self.input_feat_len = 128

        # build the ensemble
        self.ensemble = [FFEnsembleMember(input_feat_len=self.input_feat_len) for _ in range(self.num_ensemble_members)]

        
    def fit(self, x:torch.Tensor, y:torch.Tensor, conf_train:Config)->None:
        ''' Trains each ensemble member with the data
        
        Args:
            x: torch.Tensor, architecture features [Nxfeatures]
            y: torch.Tensor [Nx1] 
            the first entry are architecture features, one per row
            the second entry is the accuracy values per feature
        '''

        lr = conf_train['lr']
        num_steps = conf_train['num_steps']

        x = x.cuda()
        y = y.cuda()

        # TODO: should we be splitting data into 
        # train and val?
        for idx, member in enumerate(self.ensemble):
            logger.pushd(f'train ensemble_{idx}')
            criterion = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD(member.parameters(), lr=lr)
            member.train()
            for t in num_steps:
                y_pred = member(x)
                loss = criterion(y_pred, y)
                if t % 10 == 9:
                    logger.info(f'step {t}: training loss {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.popd()


    def predict(self, x:torch.Tensor)->MeanVar:
        ''' x: torch.Tensor architecture feature '''
        
        preds = []
        for member in self.ensemble:
            member.eval()
            preds.append(member(x).item())
        preds = np.array(preds)
        mean = np.mean(preds)
        std = np.std(preds)
        var = std * std

        return MeanVar(mean=mean, var=var)


            








class FFEnsembleMember(nn.Module):
    def __init__(self, input_feat_len:int):
        super(FFEnsembleMember, self).__init__()

        self.input_feat_len = input_feat_len

        self.net = nn.Sequential(
            nn.Linear(self.input_feat_len, self.input_feat_len*3),
            nn.ReLU(),
            nn.Linear(self.input_feat_len*3, self.input_feat_len*3),
            nn.ReLU(),
            nn.Linear(self.input_feat_len*3, self.input_feat_len),
            nn.ReLU(),
            nn.Linear(self.input_feat_len, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)