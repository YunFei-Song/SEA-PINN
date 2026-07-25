"""Shared utilities for deterministic SEA-PINN experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List

import deepxde as dde
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = ROOT / "outputs"


def set_global_seed(seed: int) -> None:
    """Match the seed handling used in the original experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dde.config.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_deepxde() -> None:
    """Use the original numerical precision."""

    dde.config.set_default_float("float64")


def parse_seed_range(value: str) -> List[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.linalg.norm(y_true)
    if denom == 0:
        return float(np.linalg.norm(y_true - y_pred))
    return float(np.linalg.norm(y_true - y_pred) / denom)
