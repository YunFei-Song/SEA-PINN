"""SEA-PINN network architecture."""

from __future__ import annotations

from typing import List

import torch

from deepxde import config
from deepxde.nn import activations
from deepxde.nn.pytorch.nn import NN


class SEAPINN(NN):
    """Neuron-wise self-adaptive weighted network used as SEA-PINN.
    """

    def __init__(self, layer_sizes: List[int], activation: str):
        super().__init__()
        if isinstance(activation, list):
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)

        self.linears = torch.nn.ModuleList()
        self.weight_generators = torch.nn.ModuleList()

        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                    )
                )
            torch.nn.init.uniform_(self.linears[-1].weight, a=-0.5, b=0.5)
            torch.nn.init.zeros_(self.linears[-1].bias)

            if i < len(layer_sizes) - 1:
                hidden = layer_sizes[i]
                generator = torch.nn.Sequential(
                    torch.nn.Linear(hidden, hidden // 4, dtype=config.real(torch)),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden // 4, hidden, dtype=config.real(torch)),
                    torch.nn.Sigmoid(),
                )
                for layer in generator:
                    if isinstance(layer, torch.nn.Linear):
                        torch.nn.init.uniform_(layer.weight, a=-0.5, b=0.5)
                        torch.nn.init.zeros_(layer.bias)
                self.weight_generators.append(generator)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for j, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            x = self.activation[j](x) if isinstance(self.activation, list) else self.activation(x)
            x = x * self.weight_generators[j](x)

        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
