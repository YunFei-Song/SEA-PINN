"""Run three-case initial-loss statistics with fixed samples and varying initialization."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_MODELS = ["FNN-PINN", "SEA-PINN", "TSA-PINN"]
VALID_CASES = [5, 7, 11]
VARIANCE_STEP = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Run 1000-seed initial-loss experiment for cases 5, 7, and 11.")
    parser.add_argument("--seeds", default="1-1000")
    parser.add_argument("--cases", nargs="+", type=int, choices=VALID_CASES, default=VALID_CASES)
    parser.add_argument("--models", nargs="+", choices=VALID_MODELS, default=VALID_MODELS)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--save-logs", action="store_true", help="Save one training log per case/model/seed run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < VARIANCE_STEP:
        raise ValueError(f"Experiment 1 requires --iterations >= {VARIANCE_STEP} for the variance plot.")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("DDE_BACKEND", "pytorch")

    from reproduction.pdes import create_case
    from reproduction.plotting import plot_initial_loss_outputs
    from reproduction.sample_data import load_training_sample
    from reproduction.training import total_train_loss_at_step, train_once
    from reproduction.utils import OUTPUT_DIR, configure_deepxde, parse_seed_range

    configure_deepxde()
    seeds = parse_seed_range(args.seeds)
    output_dir = OUTPUT_DIR / "initial_loss"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "initial_loss_case5_case7_case11.csv"
    fields = ["case_id", "seed"] + args.models + [f"{model}_step{VARIANCE_STEP}" for model in args.models]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case_id in args.cases:
            train_fixed = load_training_sample(case_id)
            for seed in seeds:
                row = {"case_id": case_id, "seed": seed}
                for model_name in args.models:
                    print(f"Initial-loss run: case={case_id}, init_seed={seed}, model={model_name}")
                    pde = create_case(case_id)
                    log_path = (
                        output_dir / "logs" / f"case{case_id}_{model_name}_seed{seed}.log"
                        if args.save_logs
                        else None
                    )
                    _, _, loss_history, _ = train_once(
                        pde=pde,
                        model_name=model_name,
                        iterations=args.iterations,
                        fixed=train_fixed,
                        init_seed=seed,
                        log_path=log_path,
                    )
                    row[model_name] = f"{total_train_loss_at_step(loss_history, 0):.12e}"
                    row[f"{model_name}_step{VARIANCE_STEP}"] = (
                        f"{total_train_loss_at_step(loss_history, VARIANCE_STEP):.12e}"
                    )
                writer.writerow(row)
                f.flush()

    plot_initial_loss_outputs(csv_path, output_dir, variance_step=VARIANCE_STEP)
    print(f"Done. Initial-loss outputs are in: {output_dir}")


if __name__ == "__main__":
    main()
