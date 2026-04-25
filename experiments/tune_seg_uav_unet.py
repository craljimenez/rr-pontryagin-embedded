"""Bayesian HPO for UNet segmentation models.

Search spaces:
    vanilla     : lr, weight_decay, unet_depth
    euclidean   : lr, weight_decay, unet_depth
    hyperbolic  : lr, weight_decay, unet_depth, hyperbolic_c
    pontryagin  : lr, weight_decay, unet_depth, kappa, d_poly,
                  rff_multiplier, lambda_topo, lambda_balance

Usage:
    python tune_seg_uav_unet.py --model pontryagin --n-calls 30 --trial-epochs 12
"""

import argparse
import csv
import json
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
from configs.seg_uav_unet import AVAILABLE_MODELS, DEVICE, RESULTS_DIR
from run_seg_uav_unet import build_dataloaders, train_one_model


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────────────────────

_COMMON = [
    Real(1e-5, 1e-2, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
    Integer(3, 5, name="unet_depth"),
]


def search_space(model_type: str):
    if model_type in ("vanilla", "euclidean"):
        return list(_COMMON)
    if model_type == "hyperbolic":
        return _COMMON + [
            Real(0.1, 2.0, prior="log-uniform", name="hyperbolic_c"),
        ]
    if model_type == "pontryagin":
        return _COMMON + [
            Integer(1, 64, prior="log-uniform", name="kappa"),
            Integer(1, 10, name="d_poly"),
            Integer(1, 4,  name="rff_multiplier"),
            Real(1e-6, 0.2, prior="log-uniform", name="lambda_topo"),
            Real(1e-3, 1.0, prior="log-uniform", name="lambda_balance"),
        ]
    raise ValueError(f"Unknown model {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  choices=list(AVAILABLE_MODELS), required=True)
    ap.add_argument("--n-calls",      type=int, default=30)
    ap.add_argument("--n-random",     type=int, default=10)
    ap.add_argument("--trial-epochs", type=int, default=12)
    ap.add_argument("--device",       default=DEVICE)
    ap.add_argument("--data-root",    type=str, default=None,
                    help="Path to UAV_segmantation dataset root (overrides config).")
    ap.add_argument("--results-dir",  type=str, default=None,
                    help="Output directory for results (overrides config).")
    args = ap.parse_args()

    from pathlib import Path as _Path
    results_dir = _Path(args.results_dir) if args.results_dir else RESULTS_DIR
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hpo_dir = results_dir / args.model / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    data  = build_dataloaders(data_root=args.data_root)
    space = search_space(args.model)

    trials_path = hpo_dir / "trials.csv"
    trial_count = [0]

    with open(trials_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([dim.name for dim in space] + ["val_macro_iou", "elapsed_s"])

    @use_named_args(space)
    def objective(**params):
        trial_count[0] += 1
        t0 = time.time()
        print(f"\n── Trial {trial_count[0]} ──────────────────────────")
        print("  params:", {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in params.items()})
        result = train_one_model(
            args.model,
            params=params,
            data=data,
            device=device,
            epochs=args.trial_epochs,
            out_dir=hpo_dir / f"trial_{trial_count[0]:03d}",
            verbose=False,
            save_viz=False,
        )
        val_iou = result["global"]["macro_iou"]
        elapsed = time.time() - t0
        print(f"  val_mIoU={val_iou:.4f}  ({elapsed:.0f}s)")

        with open(trials_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [params[dim.name] for dim in space] + [val_iou, round(elapsed)]
            )
        return -val_iou   # minimise negative IoU

    result = forest_minimize(
        objective,
        space,
        n_calls=args.n_calls,
        n_random_starts=args.n_random,
        random_state=42,
        verbose=False,
    )

    best_params = {dim.name: val for dim, val in zip(space, result.x)}
    best_iou    = -result.fun

    print(f"\n{'='*60}")
    print(f"Best val mIoU: {best_iou:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_params["_best_val_macro_iou"] = best_iou
    with open(hpo_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    skopt_dump(result, hpo_dir / "skopt_result.pkl", store_objective=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_convergence(result, ax=ax)
    ax.set_title(f"HPO convergence — UNet {args.model}")
    fig.tight_layout()
    fig.savefig(hpo_dir / "convergence.png", dpi=120)
    plt.close(fig)

    print(f"\nResults saved to {hpo_dir}")


if __name__ == "__main__":
    main()
