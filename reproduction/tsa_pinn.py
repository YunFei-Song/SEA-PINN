"""Optional TSA-PINN baseline network."""

from __future__ import annotations

from typing import List

import torch

from deepxde import config
from deepxde.nn.pytorch.nn import NN


class TSAPINN(NN):
    """PyTorch/DeepXDE implementation of the TSA-PINN baseline."""

    def __init__(self, layer_sizes: List[int], initial_frequency: float = 1.0):
        super().__init__()
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must be a list of at least two integers.")

        self.linears = torch.nn.ModuleList()
        self.freqs = torch.nn.ParameterList()

        for i in range(len(layer_sizes) - 1):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i], layer_sizes[i + 1], dtype=config.real(torch)
                )
            )
            torch.nn.init.uniform_(self.linears[-1].weight, a=-0.5, b=0.5)
            torch.nn.init.zeros_(self.linears[-1].bias)

            if i < len(layer_sizes) - 2:
                self.freqs.append(
                    torch.nn.Parameter(
                        torch.full(
                            (1, layer_sizes[i + 1]),
                            float(initial_frequency),
                            dtype=config.real(torch),
                        )
                    )
                )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for i in range(len(self.linears) - 1):
            layer = self.linears[i]
            weighted_input = torch.matmul(x, layer.weight.t())
            biased_output = self.freqs[i] * weighted_input + layer.bias
            x = 0.5 * (torch.sin(biased_output) + torch.cos(biased_output))

        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


def add_tsa_regularization(model, net) -> None:
    """Add the auxiliary frequency regularization used in the original trainer."""

    def regularization_loss_fn():
        if not hasattr(net, "freqs") or not net.freqs:
            return torch.tensor(0.0, device=next(net.parameters()).device)

        exp_mean_freqs = []
        for freq in net.freqs:
            if freq.numel() > 0:
                exp_mean_freqs.append(torch.exp(torch.mean(freq)))
        if not exp_mean_freqs:
            return torch.tensor(0.0, device=next(net.parameters()).device)

        denominator = torch.sum(torch.stack(exp_mean_freqs))
        if denominator > 1e-9:
            return 1.0 / denominator
        return torch.tensor(float("inf"), device=denominator.device)

    if not hasattr(model, "auxiliary_losses"):
        model.auxiliary_losses = []
    model.auxiliary_losses.append(regularization_loss_fn)
