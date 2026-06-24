
import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn


class ConvLSTM3DCell(nn.Module):
    """A ConvLSTM3D cell for spatiotemporal feature extraction."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.Wxi = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Whi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Wxf = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Whf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Wxc = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Whc = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Wxo = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding="same", bias=True)
        self.Who = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding="same", bias=True)

        self.Wci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wcf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wco = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.Wxf.bias is not None:
            nn.init.constant_(self.Wxf.bias, 1.0)
        if self.Whf.bias is not None:
            nn.init.constant_(self.Whf.bias, 1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        g = torch.tanh(self.Wxc(x) + self.Whc(h))
        c_next = f * c + i * g
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + c_next * self.Wco)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

