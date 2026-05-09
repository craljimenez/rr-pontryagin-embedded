"""Bayesian HPO for ConvNeXtV2-Tiny classification on Sugarcane Leaf Disease.

Optimises weighted F1 on the test set (same convention as tune_seg_uav_unet).

Search spaces:
    euclidean   : lr, lr_backbone, weight_decay
    pontryagin        : lr, lr_backbone, weight_decay,
                        srf_multiplier, d_poly, rff_multiplier,
                        lambda_topo, lambda_balance
    pontryagin_margin : lr, lr_backbone, weight_decay,
                        srf_multiplier, d_poly, rff_multiplier,
                        lambda_balance, lambda_margin, lambda_orth_W, margin

Usage:
    python tune_cls_sugarcane.py --model pontryagin        --n-calls 30 --trial-epochs 10
    python tune_cls_sugarcane.py --model pontryagin_margin --n-calls 30 --trial-epochs 10
    python tune_cls_sugarcane.py --model euclidean         --n-calls 20 --trial-epochs 10
    python tune_cls_sugarcane.py --model all               --n-calls 30 --trial-epochs 10
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from skopt import dump as skopt_dump
from skopt import forest_minimize
from skopt.plots import plot_convergence
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.cls_sugarcane import AVAILABLE_MODELS, DEVICE, RESULTS_DIR
from run_cls_sugarcane import (
    _find_dataset_root, build_dataloaders, download_dataset, train_one_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────────────────────

_COMMON = [
    Real(1e-5, 1e-2, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-3, prior="log-uniform", name="lr_backbone"),
    Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
]


_PONTRYAGIN_ARCH = [
    Real(0.01, 0.8, prior="log-uniform", name="srf_multiplier"),
    Integer(1,  6,                       name="d_poly"),
    Integer(1,  4,                       name="rff_multiplier"),
    Real(1e-3, 1.0, prior="log-uniform", name="lambda_balance"),
]


def search_space(model_type: str) -> list:
    if model_type == "euclidean":
        return list(_COMMON)
    if model_type == "pontryagin":
        return _COMMON + _PONTRYAGIN_ARCH + [
            Real(1e-6, 0.5, prior="log-uniform", name="lambda_topo"),
        ]
    if model_type == "pontryagin_margin":
        return _COMMON + _PONTRYAGIN_ARCH + [
            Real(1e-3, 2.0, prior="log-uniform", name="lambda_margin"),
            Real(1e-3, 1.0, prior="log-uniform", name="lambda_orth_W"),
            Real(0.1,  5.0,                      name="margin"),
        ]
    raise ValueError(f"Unknown model type: {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# HPO for a single model type
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    model_type: str,
    data: dict,
    device: torch.device,
    hpo_dir: Path,
    n_calls: int,
    n_random: int,
    trial_epochs: int,
):
    hpo_dir.mkdir(parents=True, exist_ok=True)
    space = search_space(model_type)

    trials_path = hpo_dir / "trials.csv"
    with open(trials_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            [dim.name for dim in space] + ["weighted_f1", "elapsed_s"]
        )

    trial_count = [0]

    @use_named_args(space)
    def objective(**params):
        trial_count[0] += 1
        t0 = time.time()
        print(f"\n── Trial {trial_count[0]} / {n_calls} ──────────────────────────")
        print("  params:", {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in params.items()})

        result = train_one_model(
            model_type,
            data=data,
            params=params,
            device=device,
            epochs=trial_epochs,
            out_dir=hpo_dir / f"trial_{trial_count[0]:03d}",
            verbose=False,
            save_viz=False,
        )

        score   = result["weighted_f1"]
        elapsed = time.time() - t0
        print(f"  weighted_f1={score:.4f}  accuracy={result['accuracy']:.4f}  "
              f"({elapsed:.0f}s)")

        with open(trials_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [params[dim.name] for dim in space] + [score, round(elapsed)]
            )

        return -score   # minimise negative weighted F1

    result = forest_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random,
        random_state=42,
        verbose=False,
    )

    # ── Save best params ─────────────────────────────────────────────────────
    best_params = {
        dim.name: (int(val) if isinstance(dim, Integer) else float(val))
        for dim, val in zip(space, result.x)
    }
    best_score = float(-result.fun)

    print(f"\n{'='*60}")
    print(f"Best weighted F1: {best_score:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_params["_best_weighted_f1"] = best_score
    with open(hpo_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    skopt_dump(result, hpo_dir / "skopt_result.pkl", store_objective=False)

    # ── Convergence plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_convergence(result, ax=ax)
    ax.set_title(f"HPO convergence — {model_type} (ConvNeXtV2-Tiny)")
    ax.set_ylabel("−weighted F1")
    fig.tight_layout()
    fig.savefig(hpo_dir / "convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Partial dependence plot (if enough trials) ───────────────────────────
    if n_calls >= 10 and len(space) > 1:
        try:
            from skopt.plots import plot_objective
            fig2, _ = plt.subplots(
                len(space), len(space),
                figsize=(3 * len(space), 3 * len(space)),
            )
            plot_objective(result, fig=fig2, size=3)
            plt.suptitle(f"Partial dependence — {model_type}", fontsize=12)
            fig2.tight_layout()
            fig2.savefig(hpo_dir / "partial_dependence.png", dpi=100, bbox_inches="tight")
            plt.close(fig2)
        except Exception:
            pass

    print(f"\nResults saved to {hpo_dir}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    ap = argparse.ArgumentParser(
        description="Bayesian HPO for ConvNeXtV2-Tiny Sugarcane Leaf Disease classifiers"
    )
    ap.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"],
                    required=True)
    ap.add_argument("--n-calls",      type=int, default=30,
                    help="Total number of HPO evaluations (default 30).")
    ap.add_argument("--n-random",     type=int, default=10,
                    help="Random initial evaluations before Bayesian fitting (default 10).")
    ap.add_argument("--trial-epochs", type=int, default=10,
                    help="Epochs per HPO trial (default 10, less than full training).")
    ap.add_argument("--device",       default=DEVICE)
    ap.add_argument("--data-root",    type=str, default=None,
                    help="Path to dataset root (downloads if omitted).")
    ap.add_argument("--results-dir",  type=str, default=None,
                    help="Output directory (overrides config).")
    return ap.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    # Resolve dataset root once
    if args.data_root:
        data_root = _find_dataset_root(Path(args.data_root))
    else:
        env_root = os.environ.get("SUGARCANE_DATA_ROOT")
        if env_root:
            data_root = _find_dataset_root(Path(env_root))
        else:
            data_root = _find_dataset_root(download_dataset())

    print(f"Dataset root: {data_root}")
    data = build_dataloaders(data_root)
    print(f"Classes: {data['classes']}")
    print(f"Train: {len(data['train_ds'])}  Val: {len(data['val_ds'])}  "
          f"Test: {len(data['test_ds'])}")

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]

    for mt in models:
        print(f"\n{'='*60}")
        print(f"HPO for: {mt.upper()}")
        print(f"  n_calls={args.n_calls}  n_random={args.n_random}  "
              f"trial_epochs={args.trial_epochs}")
        hpo_dir = results_dir / mt / "hpo"
        tune_model(
            model_type=mt,
            data=data,
            device=device,
            hpo_dir=hpo_dir,
            n_calls=args.n_calls,
            n_random=args.n_random,
            trial_epochs=args.trial_epochs,
        )

    print("\nAll HPO done.")


if __name__ == "__main__":
    main()
