"""Pre-generated sample loading helpers for the reproduction run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .pdes import create_case
from .utils import SAMPLES_DIR


def training_sample_path(case_id: int) -> Path:
    return SAMPLES_DIR / f"case{case_id}_train_samples.npz"


def evaluation_sample_path(case_id: int) -> Path:
    return SAMPLES_DIR / f"case{case_id}_eval_samples.npz"


def load_training_sample(case_id: int) -> Dict[str, np.ndarray]:
    path = training_sample_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Training sample file not found: {path}. "
            "Restore the released data/samples files before running the experiments."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_evaluation_sample(case_id: int) -> Dict[str, np.ndarray]:
    if case_id == 7:
        path = evaluation_sample_path(case_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation sample file not found: {path}. "
                "Restore the released data/samples files before running the experiments."
            )
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    if case_id in (5, 11):
        pde = create_case(case_id)
        eval_x = pde.ref_data[:, : pde.input_dim]
        eval_y = pde.ref_data[:, pde.input_dim :]
        return {
            "eval_x": eval_x,
            "eval_y": eval_y,
            "heatmap_x": eval_x,
            "heatmap_y": eval_y,
        }

    raise ValueError(f"Unsupported case_id={case_id}. This release includes cases 5, 7, and 11.")


def apply_training_sample(data, fixed: Dict[str, np.ndarray]) -> None:
    """Replace temporary DeepXDE arrays with the provided fixed arrays."""

    data.train_x_all = fixed["train_x_all"]
    data.train_x_bc = fixed["train_x_bc"]
    data.train_x = fixed["train_x"]
    data.train_y = None
    data.train_aux_vars = None
    data.num_bcs = [int(v) for v in fixed["num_bcs"]]
    data.test_x = fixed["deepxde_test_x"]
    data.test_y = None
    data.test_aux_vars = None
