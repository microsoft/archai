import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d


# feat_size: 64 for natsbench cifar10
# feat_size:  


class AddonNN(nn.Module):
    def __init__(self, model:nn.Module, feat_size:int, num_classes:int=10)->None:
        super(AddonNN, self).__init__()

        # keep everything but the last layer 
        self.featurizer = nn.Sequential(*list(model.classifier.children())[:-1])
        
        # freeze the featurizer
        for param in self.featurizer.parameters():
            param.requires_grad_ = False

        # create small network that will take features as input
        # TODO: got to figure out what is input size
        self.addon_nn = nn.Sequential(
            nn.Linear(feat_size, 4096), 
            nn.ReLU(), 
            nn.Linear(4096, 4096), 
            nn.ReLU(), 
            nn.Linear(4096, num_classes)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass through the original network up to the penultimate layer
        feats = self.featurizer(x)
        feats = torch.flatten(feats, start_dim=1) # don't flatten the batch dimension!
        # then pass through the small network
        x = self.addon_nn(feats)

