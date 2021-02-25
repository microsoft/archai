import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from archai.algos.nasbench101.model import Network

class AddonNN(nn.Module):
    def __init__(self, model:Network, num_classes:int=10, stack_num=1)->None:
        super(AddonNN, self).__init__()

        assert stack_num in set((1,2,3))

        self.FEAT_SIZE = {1: 32768, 2: 16384, 3: 32768}
        
        self.stack_num = stack_num
        self.featurizer = model

        # freeze the featurizer
        for param in self.featurizer.parameters():
            param.requires_grad = False

        # create small network that will take features as input
        self.addon_nn = nn.Linear(self.FEAT_SIZE[stack_num], num_classes)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass through the original network using non standard
        # forward call which returns intermediate activations

        feats = self.featurizer.forward_stack_1or2or3(x, stack_num=self.stack_num)
        feats = torch.flatten(feats, start_dim=1) # don't flatten the batch dimension!

        # then pass through the small network

        # HACK: since tw.ModelStats probably puts the network 
        # on cpu while the tensors are on gpu
        # TODO: Fix 
        self.addon_nn.to(torch.device('cuda'))

        x = self.addon_nn(feats)
        return x