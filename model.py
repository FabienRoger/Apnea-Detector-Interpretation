import torch
import torch.nn as nn

from constants import *


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels=10,
        out_channels=10,
        kernel_size=5,
        pooling_size=3,
        padding="valid",
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.maxPool = nn.MaxPool1d(pooling_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        activation = x
        x = self.maxPool(x)
        x = self.activation(x)
        return x, activation


class Simple1DCNN_Open(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                ConvLayer(in_channels=N_signals),
                ConvLayer(),
            ]
        )
        self.flatten = nn.Flatten()
        self.final_lin = nn.Linear(9 * 10, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x, extract=False):
        activations = [x]

        for layer in self.conv_layers:
            x, activation = layer(x)
            activations.append(activation)

        x = self.flatten(x)
        activations.append(x)

        x = self.final_lin(x)
        x = self.final_act(x)
        activations.append(x)

        x = torch.reshape(x, (-1,))

        if extract:
            return x, activations
        else:
            return x
