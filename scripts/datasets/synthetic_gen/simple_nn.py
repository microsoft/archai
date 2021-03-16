import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, num_classes:int=10):
        super(SimpleNN, self).__init__()

        self.net = nn.Sequential(
                                nn.Linear(32*32*3, 3072),
                                nn.ReLU(),                                                                                        
                                nn.Linear(3072, 1)
                            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.net(x)