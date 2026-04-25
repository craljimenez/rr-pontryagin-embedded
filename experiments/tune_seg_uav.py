"""Bayesian hyperparameter search for UAV segmentation.

Uses scikit-optimize's `forest_minimize` (Extra Trees surrogate), matching the
style used in ../002_rff_ss. Search spaces are model-specific:

    euclidean  : lr, weight_decay, backbone_depth
    hyperbolic : lr, weight_decay, c, backbone_depth
    pontryagin : lr, weight_decay, kappa, d_poly, rff_multiplier,
                 lambda_topo, lambda_balance, backbone_depth

Tied by design (not tuned):
    n_srf = kappa                         (negative subspace dim = signature)
    n_rff = rff_multiplier * in_channels  (positive subspace width)

The `batch_size` and `backbone_out_ch` are held fixed, while backbone depth is
also tunable so the architecture can adapt across all heads.

    python tune_seg_uav.py --model pontryagin --n-calls 30 --trial-epochs 12

Outputs under results/seg_uav/<model>/hpo/:
    best_params.json   — best (lr, …) found
    trials.csv         — every trial's params + score
    convergence.png    — skopt convergence curve
    skopt_result.pkl   — full skopt OptimizeResult (for later inspection)
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
from configs.seg_uav import AVAILABLE_MODELS, DEVICE, RESULTS_DIR
from run_seg_uav import build_dataloaders, train_one_model


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_SPACE = [
    Real(1e-5, 1e-2, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
    Integer(3, 5, name="backbone_depth"),
]


def search_space(model_type: str):
    if model_type == "euclidean":
        return list(_COMMON_SPACE)
    if model_type == "hyperbolic":
        return _COMMON_SPACE + [
            Real(0.1, 2.0, prior="log-uniform", name="hyperbolic_c"),
        ]
    if model_type == "pontryagin":
        # `sigma` of the RFF is gradient-tuned inside the layer (log_sigma),
        # so it is no longer part of the HPO search.
        # `kappa`  = Pontryagin signature κ = N_SRF (number of negative dims in J)
        # `d_poly` = polynomial degree of K_-(x,y)=(xᵀy)^{d_poly}, independent of κ
        return _COMMON_SPACE + [
            Integer(1, 64, prior="log-uniform", name="kappa"),
            Integer(1, 10, name="d_poly"),
            Integer(1, 4, name="rff_multiplier"),
            Real(1e-6, 0.2, prior="log-uniform", name="lambda_topo"),
            Real(1e-3, 1.0, prior="log-uniform", name="lambda_balance"),
        ]
    raise ValueError(f"Unknown model {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=AVAILABLE_MODELS, required=True)
    p.add_argument("--n-calls", type=int, default=30,
                   help="Total number of trials.")
    p.add_argument("--n-random-starts", type=int, default=10,
                   help="Random initialisation trials before surrogate kicks in.")
    p.add_argument("--trial-epochs", type=int, default=12,
                   help="Epochs per trial (short schedule).")
    p.add_argument("--trial-patience", type=int, default=6,
                   help="Early-stop patience inside each trial.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-estimator", default="ET", choices=["ET", "RF"],
                   help="Surrogate model for forest_minimize.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to UAV_segmantation dataset root (overrides config).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from pathlib import Path as _Path
    results_dir = _Path(args.results_dir) if args.results_dir else RESULTS_DIR

    hpo_dir = results_dir / args.model / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Data is built once and reused across trials.
    data = build_dataloaders(data_root=args.data_root)
    print(f"Train: {len(data['train_ds'])} | Valid: {len(data['valid_ds'])} "
          f"| Test: {len(data['test_ds'])}")

    space = search_space(args.model)
    print(f"\nSearch space for {args.model}:")
    for dim in space:
        print(f"  {dim.name:<16} → {dim}")

    trials_path = hpo_dir / "trials.csv"
    trial_fieldnames = ["trial", "score"] + [d.name for d in space] + [
        "best_val_macro_iou", "seconds",
    ]
    with open(trials_path, "w", newline="") as f:
        csv.writer(f).writerow(trial_fieldnames)

    trial_counter = {"n": 0}

    @use_named_args(space)
    def objective(**raw_params):
        trial_counter["n"] += 1
        trial_id = trial_counter["n"]

        # Cast each param based on its declared Space type
        params = {}
        for dim in space:
            v = raw_params[dim.name]
            params[dim.name] = int(v) if isinstance(dim, Integer) else float(v)

        trial_dir = hpo_dir / f"trial_{trial_id:03d}"
        print(f"\n{'═' * 60}")
        print(f"Trial {trial_id:>3} / {args.n_calls}  "
              f"({args.model})  →  {trial_dir.name}")
        for k, v in params.items():
            if isinstance(v, int):
                print(f"  {k} = {v}")
            else:
                print(f"  {k} = {v:.5g}")
        print("═" * 60)

        t0 = time.time()
        try:
            result = train_one_model(
                model_type=args.model,
                params=params,
                out_dir=trial_dir,
                epochs=args.trial_epochs,
                patience=args.trial_patience,
                device=device,
                data=data,
                save_viz=False,
                verbose=False,
            )
            score = result["best_val_macro_iou"]
        except Exception as e:
            print(f"  ⚠  trial failed: {e}")
            score = 0.0
            result = {"best_val_macro_iou": 0.0}
        dt = time.time() - t0

        # Append to trials.csv
        with open(trials_path, "a", newline="") as f:
            row = [trial_id, round(score, 6)]
            row += [params[d.name] for d in space]
            row += [round(result["best_val_macro_iou"], 6), round(dt, 1)]
            csv.writer(f).writerow(row)

        print(f"  → val macro mIoU = {score:.4f}  ({dt:.1f}s)")
        # skopt minimizes
        return -score

    # ── Run optimisation ──────────────────────────────────────────────────────
    print(f"\nStarting forest_minimize ({args.n_calls} trials, "
          f"{args.n_random_starts} random) …")
    result = forest_minimize(
        func=objective,
        dimensions=space,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts,
        base_estimator=args.base_estimator,
        random_state=args.seed,
        verbose=True,
    )

    # ── Save best params ──────────────────────────────────────────────────────
    best_params = {}
    for dim, v in zip(space, result.x):
        best_params[dim.name] = int(v) if isinstance(dim, Integer) else float(v)
    best_params["_best_val_macro_iou"] = float(-result.fun)

    with open(hpo_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    # skopt.dump strips the non-picklable objective before saving
    skopt_dump(result, hpo_dir / "skopt_result.pkl", store_objective=False)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_convergence(result, ax=ax)
    ax.set_title(f"HPO convergence — {args.model}")
    fig.tight_layout()
    fig.savefig(hpo_dir / "convergence.png", dpi=120)
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"Search finished — best val macro mIoU = {-result.fun:.4f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        if k.startswith("_"):
            continue
        print(f"  {k} = {v}")
    print(f"\nResults saved under: {hpo_dir}")
    print(f"Next step:  python run_seg_uav.py --model {args.model} --load-best-params")


if __name__ == "__main__":
    main()
