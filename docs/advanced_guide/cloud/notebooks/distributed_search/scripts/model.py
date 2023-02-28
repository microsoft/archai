import json
from torch import nn

class MyModel(nn.Module):
    def __init__(self, nb_layers: int = 5, kernel_size: int = 3, hidden_dim: int = 32):
        super().__init__()

        self.nb_layers = nb_layers
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        layer_list = []

        for i in range(nb_layers):
            in_ch = (1 if i == 0 else hidden_dim)

            layer_list += [
                nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ]

        layer_list += [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(hidden_dim, 10, kernel_size=1)
        ]

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x).squeeze()

    def get_archid(self):
        return f'({self.nb_layers}, {self.kernel_size}, {self.hidden_dim})'

    def save_config(self, filename):
        with open(filename, 'w') as fp:
            json.dump({
                'nb_layers': self.nb_layers,
                'kernel_size': self.kernel_size,
                'hidden_dim': self.hidden_dim
            }, fp)

    @staticmethod
    def load_from_config(self, filename):
        config = json.load(open(filename))
        nb_layers = int(config['nb_layers'])
        kernel_size = int(config['kernel_size'])
        hidden_dim = int(config['hidden_dim'])
        return MyModel(nb_layers, kernel_size, hidden_dim)
