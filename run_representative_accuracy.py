"""Run representative accuracy experiments with fixed samples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


VALID_MODELS = ["FNN-PINN", "SEA-PINN", "TSA-PINN"]
VALID_CASES = [5, 7, 11]


def parse_args():
    parser = argparse.ArgumentParser(description="SEA-PINN representative accuracy experiment")
    parser.add_argument("--models", nargs="+", choices=VALID_MODELS, default=["FNN-PINN", "SEA-PINN"])
    parser.add_argument("--cases", nargs="+", type=int, choices=VALID_CASES, default=VALID_CASES)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Use 100 iterations unless --iterations is also set.")
    parser.add_argument("--init-seed", type=int, default=42, help="Network initialization seed.")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("DDE_BACKEND", "pytorch")

    from reproduction.sample_data import load_evaluation_sample, load_training_sample
    from reproduction.pdes import create_case
    from reproduction.plotting import plot_loss_comparison, plot_solution_heatmap
    from reproduction.training import evaluate, save_train_loss_csv, train_once, write_summary
    from reproduction.utils import OUTPUT_DIR, configure_deepxde, ensure_dirs

    configure_deepxde()
    iterations = 100 if args.quick and args.iterations == 5000 else args.iterations
    output_dir = OUTPUT_DIR / "representative_accuracy"
    loss_dir = output_dir / "loss"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    ensure_dirs([loss_dir, figure_dir, log_dir])

    summary_rows = []
    for case_id in args.cases:
        train_fixed = load_training_sample(case_id)
        eval_fixed = load_evaluation_sample(case_id)
        case_loss_files = {}
        predictions_for_plot = {}
        pde_for_plot = None
        for model_name in args.models:
            print(
                f"Running case {case_id}, model {model_name}, "
                f"iterations={iterations}, init_seed={args.init_seed}"
            )
            pde = create_case(case_id)
            model, _, loss_history, runtime = train_once(
                pde=pde,
                model_name=model_name,
                iterations=iterations,
                fixed=train_fixed,
                init_seed=args.init_seed,
                log_path=log_dir / f"case{case_id}_{model_name}.log",
                show_progress=True,
                progress_label=f"case {case_id} {model_name}",
            )
            loss_path = loss_dir / f"case{case_id}_{model_name}_loss.csv"
            save_train_loss_csv(loss_path, pde, loss_history)
            rel_l2, max_error, eval_pred = evaluate(model, eval_fixed)
            summary_rows.append(
                {
                    "case_id": case_id,
                    "case_name": pde.name,
                    "model": model_name,
                    "init_seed": args.init_seed,
                    "iterations": iterations,
                    "device": args.device,
                    "rel_l2": f"{rel_l2:.12e}",
                    "max_abs_error": f"{max_error:.12e}",
                    "runtime_sec": f"{runtime:.2f}",
                }
            )
            case_loss_files[model_name] = loss_path
            pde_for_plot = pde
            heatmap_pred = model.predict(eval_fixed["heatmap_x"])
            predictions_for_plot[model_name] = heatmap_pred
            print(f"  rel_l2={rel_l2:.4e}, max_abs_error={max_error:.4e}")

        plot_loss_comparison(case_id, case_loss_files, figure_dir / f"case{case_id}_loss_comparison")
        component_names = [item["name"] for item in pde_for_plot.output_config]
        plot_solution_heatmap(
            pde=pde_for_plot,
            case_id=case_id,
            x=eval_fixed["heatmap_x"],
            y_true=eval_fixed["heatmap_y"],
            predictions=predictions_for_plot,
            output_path=figure_dir / f"case{case_id}_heatmap",
            component_names=component_names,
        )

    write_summary(output_dir / "summary.csv", summary_rows)
    print(f"Done. Representative accuracy outputs are in: {output_dir}")


if __name__ == "__main__":
    main()
