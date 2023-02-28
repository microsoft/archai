import torch
from torch import nn

from archai.discrete_search.search_spaces.config import ArchConfig


class SeparableConv1d(nn.Module):
    def __init__(self, arch_config: ArchConfig, hidden_size: int,
                 total_heads: int, op_heads: int, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        self.op_size = op_heads * (hidden_size // total_heads)
        self.kernel_size = arch_config.pick('kernel_size')

        self.conv_map_in = nn.Linear(hidden_size, self.op_size)
        self.conv = nn.Conv1d(
            self.op_size, self.op_size, self.kernel_size,
            padding=(self.kernel_size-1), groups=self.op_size
        )
        
        self.act = nn.ReLU()
        
    def forward(self, hidden_states: torch.FloatTensor, **kwargs):
        out = self.act(self.conv_map_in(hidden_states))
        out = self.act(self.conv(out.transpose(-1,-2)).transpose(-1,-2))
        
        # Removes padding to get back the original sequence length
        out = out[:, :hidden_states.shape[1], :]
        
        return out, None
