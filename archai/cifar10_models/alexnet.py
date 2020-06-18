import torch
import torch.nn as nn
import os

__all__ = ['AlexNet','alexnet']

class AlexNet(nn.Module):
 
    def __init__(self,num_classes=1000,init_weights='True'):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=(4,4), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5,5), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool=nn.AdaptiveAvgPool2d((6, 6))
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        self.softmax=nn.Softmax(dim=1)
        if(init_weights):
            self.init_weights()

    def init_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)
        nn.init.constant_(self.features[0].bias, 1)
        nn.init.constant_(self.features[8].bias, 1)

    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.classifier(x)
        x=self.softmax(x)
        return x


def alexnet(pretrained=False, progress=True, device='cpu', **kwargs):
    """
    AlexNet architecture implemented from the paper 
    `"ImageNet Classification with Deep Convolutional Neural Networks" <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`

    Args:
        pretrained (bool): If True, returns a pre-trained model. In that case, the 'init_weights' argument of 'AlexNet' class is set to False
        progress (bool): If True, displays a progress bar of the download to stderr
        device: default is 'cpu'
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = AlexNet(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/alexnet.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model
