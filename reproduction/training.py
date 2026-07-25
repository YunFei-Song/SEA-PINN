"""Training and evaluation utilities."""

from __future__ import annotations

import contextlib
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

import deepxde as dde
import numpy as np
import torch

from .sample_data import apply_training_sample
from .networks import add_tsa_regularization, create_network
from .pdes import BaseCase
from .utils import l2_relative_error, set_global_seed


class InMemoryBestModel(dde.callbacks.Callback):
    """Keep the best train-loss state in memory."""

    def __init__(self):
        super().__init__()
        self.best_loss = np.inf
        self.best_step = 0
        self.best_state = None

    def on_epoch_end(self):
        loss_train = getattr(self.model.train_state, "loss_train", None)
        if loss_train is None:
            return
        current = float(np.sum(loss_train))
        if current < self.best_loss:
            self.best_loss = current
            self.best_step = int(self.model.train_state.step)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.net.state_dict().items()
            }

    def restore(self):
        if self.best_state is not None:
            self.model.net.load_state_dict(self.best_state)


class TerminalProgressBar(dde.callbacks.Callback):
    """Render a compact terminal progress bar for one training run."""

    def __init__(
        self,
        total_steps: int,
        label: str,
        stream: Optional[TextIO] = None,
        width: int = 28,
        min_interval: float = 0.2,
    ):
        super().__init__()
        self.total_steps = max(1, int(total_steps))
        self.label = label
        self.stream = stream or sys.stdout
        self.width = int(width)
        self.min_interval = float(min_interval)
        self.start_step = 0
        self.start_time = 0.0
        self.last_update = 0.0
        self.last_done = -1

    def on_train_begin(self):
        self.start_step = int(getattr(self.model.train_state, "step", 0))
        self.start_time = time.perf_counter()
        self.last_update = 0.0
        self.last_done = -1
        self._render(force=True)

    def on_epoch_end(self):
        self._render(force=False)

    def on_train_end(self):
        if self.last_done < self.total_steps:
            self._render(force=True)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self.last_update < self.min_interval:
            return
        current_step = int(getattr(self.model.train_state, "step", self.start_step))
        done = min(self.total_steps, max(0, current_step - self.start_step))
        fraction = done / self.total_steps
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start_time
        if done > 0 and done < self.total_steps:
            eta = elapsed * (self.total_steps - done) / done
            eta_text = f", eta {eta:6.1f}s"
        else:
            eta_text = ""
        self.stream.write(
            f"\r{self.label} [{bar}] {done:>{len(str(self.total_steps))}}/"
            f"{self.total_steps} ({fraction * 100:5.1f}%), elapsed {elapsed:6.1f}s"
            f"{eta_text}"
        )
        self.stream.flush()
        self.last_update = now
        self.last_done = done


def build_model(
    pde: BaseCase,
    model_name: str,
    fixed: Optional[Dict[str, np.ndarray]] = None,
    init_seed: Optional[int] = None,
):
    # DeepXDE's IC/BC bookkeeping does not accept fully empty TimePDE samples.
    # For fixed-sample runs we therefore construct a normal data object and
    # immediately replace every training/test array with the shipped fixed data.
    # Any sampled arrays created here are discarded before training starts.
    data = pde.make_data(sample=True)
    if fixed is not None:
        apply_training_sample(data, fixed)

    if init_seed is not None:
        set_global_seed(init_seed)
    net = create_network(model_name, pde.input_dim, pde.output_dim)
    model = dde.Model(data, net)
    model.pde = pde
    model.skip_test_loss = True

    loss_weights = [1.0] * len(pde.loss_config)
    if model_name == "TSA-PINN":
        add_tsa_regularization(model, net)
        loss_weights.append(1.0)
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    return model


def train_once(
    pde: BaseCase,
    model_name: str,
    iterations: int,
    fixed: Optional[Dict[str, np.ndarray]],
    init_seed: Optional[int] = None,
    log_path: Optional[Path] = None,
    show_progress: bool = False,
    progress_label: Optional[str] = None,
) -> Tuple[dde.Model, InMemoryBestModel, object, float]:
    progress_stream = sys.stdout

    def run_training():
        built_model = build_model(pde, model_name, fixed=fixed, init_seed=init_seed)
        built_saver = InMemoryBestModel()
        callbacks = [built_saver]
        if show_progress:
            callbacks.append(
                TerminalProgressBar(
                    total_steps=iterations,
                    label=progress_label or f"{pde.name} {model_name}",
                    stream=progress_stream,
                )
            )
        history, _ = built_model.train(
            iterations=iterations,
            display_every=1,
            callbacks=callbacks,
            save_model=False,
        )
        return built_model, built_saver, history

    start = time.perf_counter()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                model, saver, loss_history = run_training()
    else:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model, saver, loss_history = run_training()
    elapsed = time.perf_counter() - start
    saver.restore()
    return model, saver, loss_history, elapsed


def loss_headers(pde: BaseCase, n_loss_terms: int) -> List[str]:
    names = [item["name"] for item in pde.loss_config]
    if n_loss_terms > len(names):
        names += [f"auxiliary_{i + 1}" for i in range(n_loss_terms - len(names))]
    return ["Iteration"] + [f"Train Loss ({name})" for name in names] + ["Total Train Loss"]


def save_train_loss_csv(path: Path, pde: BaseCase, loss_history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    losses = [np.asarray(row, dtype=float) for row in loss_history.loss_train]
    headers = loss_headers(pde, len(losses[0]) if losses else len(pde.loss_config))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for step, row in zip(loss_history.steps, losses):
            writer.writerow([int(step)] + [f"{float(v):.12e}" for v in row] + [f"{float(np.sum(row)):.12e}"])


def evaluate(model: dde.Model, fixed: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
    x = fixed["eval_x"]
    y_true = fixed["eval_y"]
    y_pred = model.predict(x)
    return l2_relative_error(y_true, y_pred), float(np.max(np.abs(y_true - y_pred))), y_pred


def write_summary(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "case_name",
        "model",
        "init_seed",
        "iterations",
        "device",
        "rel_l2",
        "max_abs_error",
        "runtime_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_total_train_loss(loss_history) -> float:
    if not loss_history.loss_train:
        return float("nan")
    return float(np.sum(np.asarray(loss_history.loss_train[0], dtype=float)))


def total_train_loss_at_step(loss_history, target_step: int) -> float:
    if not loss_history.loss_train:
        return float("nan")

    losses = [np.asarray(row, dtype=float) for row in loss_history.loss_train]
    steps = [int(step) for step in getattr(loss_history, "steps", [])]
    for step, row in zip(steps, losses):
        if step == int(target_step):
            return float(np.sum(row))

    if target_step == 0:
        return float(np.sum(losses[0]))
    if not steps and 0 <= target_step < len(losses):
        return float(np.sum(losses[target_step]))

    available = ", ".join(str(step) for step in steps) if steps else "none"
    raise ValueError(f"Training loss at step {target_step} is not available. Available steps: {available}.")
