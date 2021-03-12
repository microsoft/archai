import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, num_classes:int=10):
        super(SimpleNN, self).__init__()

        self.net = nn.Sequential(
                                nn.Linear(32*32*3, 2048),
                                nn.ReLU(),                            
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, num_classes)
                            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.net(x)