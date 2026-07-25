"""Network factory for the SEA-PINN reproduction examples."""

from __future__ import annotations

from typing import List

from .fnn import create_fnn
from .sea_pinn import SEAPINN
from .tsa_pinn import TSAPINN, add_tsa_regularization


MODEL_NAMES = ("FNN-PINN", "SEA-PINN", "TSA-PINN")


def layer_sizes(input_dim: int, output_dim: int) -> List[int]:
    return [input_dim] + [32] * 9 + [output_dim]

def create_network(model_name: str, input_dim: int, output_dim: int):
    sizes = layer_sizes(input_dim, output_dim)
    if model_name == "FNN-PINN":
        return create_fnn(sizes)
    if model_name == "SEA-PINN":
        return SEAPINN(sizes, "silu")
    if model_name == "TSA-PINN":
        return TSAPINN(sizes, initial_frequency=1.0)
    raise ValueError(f"Unsupported model: {model_name}. Expected one of {MODEL_NAMES}.")
