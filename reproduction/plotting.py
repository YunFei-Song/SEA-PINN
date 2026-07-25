"""Plot helpers for loss curves, heatmaps, and initial-loss summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _griddata_fill(points, values, grid_x, grid_y, method="cubic"):
    vals = griddata(points, values, (grid_x, grid_y), method=method)
    if np.isnan(vals).any():
        nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
        vals = np.where(np.isnan(vals), nearest, vals)
    return vals


def _mask_geometry(pde, grid_x, grid_y, *arrays):
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    try:
        inside = pde.geom.inside(pts).reshape(grid_x.shape)
    except Exception:
        return arrays
    masked = []
    for array in arrays:
        out = np.array(array, copy=True)
        out[~inside] = np.nan
        masked.append(out)
    return tuple(masked)


def plot_loss_comparison(case_id: int, loss_files: Dict[str, Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    styles = {
        "FNN-PINN": ("#1f77b4", "-"),
        "SEA-PINN": ("#d62728", "--"),
        "TSA-PINN": ("#2ca02c", "-."),
    }
    for model_name, path in loss_files.items():
        df = pd.read_csv(path)
        color, linestyle = styles.get(model_name, ("black", "-"))
        plt.semilogy(df["Iteration"], df["Total Train Loss"], label=model_name, color=color, linestyle=linestyle)
    plt.xlabel("Iteration")
    plt.ylabel("Total training loss")
    plt.title(f"Case {case_id} loss comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def plot_solution_heatmap(
    pde,
    case_id: int,
    x: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    component_names: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for comp_idx, comp_name in enumerate(component_names):
        _plot_component_heatmap(
            pde=pde,
            case_id=case_id,
            x=x,
            y_true=y_true[:, comp_idx],
            predictions={name: pred[:, comp_idx] for name, pred in predictions.items()},
            output_path=output_path.with_name(f"{output_path.name}_{comp_name}"),
            component_name=comp_name,
        )


def _plot_component_heatmap(
    pde,
    case_id: int,
    x: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    component_name: str,
) -> None:
    points = x[:, :2]
    if case_id == 11 and hasattr(pde, "bbox"):
        bbox = pde.bbox
        grid_x, grid_y = np.mgrid[bbox[0] : bbox[1] : 220j, bbox[2] : bbox[3] : 160j]
    else:
        grid_x, grid_y = np.mgrid[
            points[:, 0].min() : points[:, 0].max() : 200j,
            points[:, 1].min() : points[:, 1].max() : 200j,
        ]

    grid_true = _griddata_fill(points, y_true, grid_x, grid_y)
    grid_preds = {name: _griddata_fill(points, pred, grid_x, grid_y) for name, pred in predictions.items()}
    if case_id == 11:
        arrays = _mask_geometry(pde, grid_x, grid_y, grid_true, *grid_preds.values())
        grid_true = arrays[0]
        grid_preds = {name: arr for name, arr in zip(grid_preds.keys(), arrays[1:])}

    n_rows = len(predictions)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.6 * n_rows), squeeze=False)
    all_solution_values = [grid_true] + list(grid_preds.values())
    solution_vmin = min(float(np.nanmin(v)) for v in all_solution_values)
    solution_vmax = max(float(np.nanmax(v)) for v in all_solution_values)
    max_error = max(float(np.nanmax(np.abs(grid_true - pred))) for pred in grid_preds.values())

    for row_idx, (model_name, grid_pred) in enumerate(grid_preds.items()):
        panels = [
            ("Analytical solution" if case_id == 7 else "Reference solution", grid_true, "viridis", solution_vmin, solution_vmax),
            (f"{model_name} prediction", grid_pred, "viridis", solution_vmin, solution_vmax),
            (f"{model_name} absolute error", np.abs(grid_true - grid_pred), "magma", 0.0, max_error),
        ]
        for col_idx, (title, values, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            mesh = ax.pcolormesh(
                grid_x,
                grid_y,
                np.ma.masked_invalid(values),
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
                edgecolors="none",
                linewidth=0,
                antialiased=False,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title} ({component_name})")
            if case_id in (5, 7, 11):
                ax.set_aspect("equal", adjustable="box")
            if case_id == 11 and hasattr(pde, "bbox"):
                ax.set_xlim(pde.bbox[0], pde.bbox[1])
                ax.set_ylim(pde.bbox[2], pde.bbox[3])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.04)
            fig.colorbar(mesh, cax=cax)

    fig.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_initial_loss_outputs(csv_path: Path, output_dir: Path, variance_step: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    candidate_models = ["FNN-PINN", "SEA-PINN", "TSA-PINN"]
    model_names = [model for model in candidate_models if model in df.columns]
    variance_columns = {
        model: f"{model}_step{variance_step}" if f"{model}_step{variance_step}" in df.columns else model
        for model in model_names
    }
    color_map = {"FNN-PINN": "#1f77b4", "SEA-PINN": "#d62728", "TSA-PINN": "#2ca02c"}

    plot_data = []
    positions = []
    labels = []
    plot_colors = []
    pos = 1
    for case_id in sorted(df["case_id"].unique()):
        case_df = df[df["case_id"] == case_id]
        for model in model_names:
            vals = case_df[model].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            plot_data.append(vals)
            positions.append(pos)
            labels.append(f"Case {case_id}\n{model}")
            plot_colors.append(color_map[model])
            pos += 1
        pos += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(plot_data, positions=positions, showmeans=False, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(plot_colors[i])
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)
    if any(np.any(vals > 0) for vals in plot_data):
        ax.set_yscale("log")
    ax.set_ylabel("Initial total training loss")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "initial_loss_violin_case5_case7_case11.png", dpi=300)
    fig.savefig(output_dir / "initial_loss_violin_case5_case7_case11.pdf")
    plt.close(fig)

    rows = []
    for case_id in sorted(df["case_id"].unique()):
        case_df = df[df["case_id"] == case_id]
        for model in model_names:
            vals = case_df[variance_columns[model]].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "loss_step": variance_step,
                    "count": len(vals),
                    "mean": float(np.mean(vals)),
                    "variance": float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    variance_df = pd.DataFrame(rows)
    variance_df.to_csv(output_dir / "initial_loss_variance_statistics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.22
    case_ids = sorted(df["case_id"].unique())
    x = np.arange(len(case_ids))
    n_models = len(model_names)
    for idx, model in enumerate(model_names):
        vals = [
            variance_df[(variance_df["case_id"] == case_id) & (variance_df["model"] == model)]["variance"].iloc[0]
            for case_id in case_ids
        ]
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=color_map[model], label=model)
    if (variance_df["variance"] > 0).any():
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {case_id}" for case_id in case_ids])
    ax.set_ylabel(f"Variance of total loss at step {variance_step}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "initial_loss_variance_bar.png", dpi=300)
    fig.savefig(output_dir / "initial_loss_variance_bar.pdf")
    plt.close(fig)
