"""Bayesian HPO for 1-way K-shot few-shot segmentation on UAV Sugarcane.

Optimises mean val IoU over N_EPISODES_VAL episodes.

Search spaces:
    euclidean   : lr, weight_decay, bce_weight, pos_weight
    pontryagin  : lr, weight_decay, bce_weight, pos_weight,
                  rff_multiplier, srf_multiplier, sigma, lambda_cone, lambda_orth

Usage:
    python tune_fss_sugarcane.py --model euclidean  --n-calls 20 --trial-epochs 10
    python tune_fss_sugarcane.py --model pontryagin --n-calls 30 --trial-epochs 10
    python tune_fss_sugarcane.py --model all        --n-calls 30 --trial-epochs 10 --k-shot 1
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
from skopt.space import Integer, Real
from skopt.utils import use_named_args

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.fss_sugarcane import (
    AVAILABLE_MODELS, DATASET_ROOT, DEVICE, K_SHOT,
    N_EPISODES_TRAIN, N_EPISODES_VAL, RESULTS_DIR,
)
from run_fss_sugarcane import train_one_model


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────────────────────

_COMMON = [
    Real(1e-5, 1e-2, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
    Real(0.1,  0.9,                       name="bce_weight"),
    Real(1.0, 10.0,  prior="log-uniform", name="pos_weight"),
]

_PONTRYAGIN_EXTRA = [
    Integer(1, 4,                         name="rff_multiplier"),  # n_rff = mult × N_RFF
    Integer(1, 4,                         name="srf_multiplier"),  # n_srf = mult × N_SRF
    Real(0.1, 5.0,  prior="log-uniform",  name="sigma"),
    Real(1e-3, 1.0, prior="log-uniform",  name="lambda_cone"),
    Real(1e-3, 0.5, prior="log-uniform",  name="lambda_orth"),
]


def search_space(model_name: str) -> list:
    if model_name == "euclidean":
        return list(_COMMON)
    if model_name == "pontryagin":
        return list(_COMMON) + list(_PONTRYAGIN_EXTRA)
    raise ValueError(f"Unknown model: {model_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# HPO for a single model
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    model_name: str,
    data_root: Path,
    device: torch.device,
    hpo_dir: Path,
    k_shot: int,
    n_calls: int,
    n_random: int,
    trial_epochs: int,
    n_episodes_train: int,
    n_episodes_val: int,
    trainable_rff: bool = False,
) -> dict:
    hpo_dir.mkdir(parents=True, exist_ok=True)
    space = search_space(model_name)

    trials_path = hpo_dir / "trials.csv"
    with open(trials_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            [dim.name for dim in space] + ["val_iou", "elapsed_s"]
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
            model_name,
            data_root=data_root,
            device=device,
            params=params,
            k_shot=k_shot,
            epochs=trial_epochs,
            out_dir=hpo_dir / f"trial_{trial_count[0]:03d}",
            verbose=False,
            save_viz=False,
            n_episodes_train=n_episodes_train,
            n_episodes_val=n_episodes_val,
            trainable_rff=trainable_rff,
        )

        score   = result["val_iou"]
        elapsed = time.time() - t0
        print(f"  val_iou={score:.4f}  ({elapsed:.0f}s)")

        with open(trials_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [params[dim.name] for dim in space] + [score, round(elapsed)]
            )

        return -score   # forest_minimize minimises

    result = forest_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random,
        random_state=42,
        verbose=False,
    )

    # ── best params ──────────────────────────────────────────────────────────
    best_params = {
        dim.name: (int(val) if isinstance(dim, Integer) else float(val))
        for dim, val in zip(space, result.x)
    }
    best_score = float(-result.fun)

    print(f"\n{'='*60}")
    print(f"Best val IoU: {best_score:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_params["_best_val_iou"] = best_score
    (hpo_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    skopt_dump(result, hpo_dir / "skopt_result.pkl", store_objective=False)

    # ── convergence plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_convergence(result, ax=ax)
    ax.set_title(f"HPO convergence — {model_name} FSS ({k_shot}-shot)")
    ax.set_ylabel("−val IoU")
    fig.tight_layout()
    fig.savefig(hpo_dir / "convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── partial dependence (skips gracefully if too few trials) ─────────────
    if n_calls >= 10 and len(space) > 1:
        try:
            from skopt.plots import plot_objective
            fig2, _ = plt.subplots(
                len(space), len(space),
                figsize=(3 * len(space), 3 * len(space)),
            )
            plot_objective(result, fig=fig2, size=3)
            plt.suptitle(f"Partial dependence — {model_name} ({k_shot}-shot)",
                         fontsize=12)
            fig2.tight_layout()
            fig2.savefig(hpo_dir / "partial_dependence.png",
                         dpi=100, bbox_inches="tight")
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
        description="Bayesian HPO for few-shot segmentation on UAV Sugarcane"
    )
    ap.add_argument("--model",
                    choices=list(AVAILABLE_MODELS) + ["all"],
                    required=True)
    ap.add_argument("--k-shot", type=int, default=K_SHOT)
    ap.add_argument("--n-calls",      type=int, default=30,
                    help="Total HPO evaluations (default 30).")
    ap.add_argument("--n-random",     type=int, default=10,
                    help="Random initialisations before Bayesian fitting (default 10).")
    ap.add_argument("--trial-epochs", type=int, default=10,
                    help="Epochs per HPO trial (default 10).")
    ap.add_argument("--n-episodes-train", type=int, default=100,
                    help="Episodes per training epoch in HPO trials (default 100).")
    ap.add_argument("--n-episodes-val",   type=int, default=50,
                    help="Episodes for val in HPO trials (default 50).")
    ap.add_argument("--device",       default=DEVICE)
    ap.add_argument("--data-root",    type=str, default=None,
                    help="Path to UAV_segmantation root (overrides config).")
    ap.add_argument("--results-dir",  type=str, default=None,
                    help="Output directory (overrides config).")
    ap.add_argument("--trainable-rff", action="store_true", default=False,
                    help="Make RFF frequencies and phases learnable; appends '_trainable' to output folder.")
    return ap.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root   = Path(args.data_root)   if args.data_root   else DATASET_ROOT
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"HPO for: {model_name.upper()}  ({args.k_shot}-shot)")
        print(f"  n_calls={args.n_calls}  n_random={args.n_random}  "
              f"trial_epochs={args.trial_epochs}")
        suffix  = "_trainable" if args.trainable_rff else ""
        hpo_dir = results_dir / f"{model_name}_{args.k_shot}shot{suffix}" / "hpo"
        tune_model(
            model_name=model_name,
            data_root=data_root,
            device=device,
            hpo_dir=hpo_dir,
            k_shot=args.k_shot,
            n_calls=args.n_calls,
            n_random=args.n_random,
            trial_epochs=args.trial_epochs,
            n_episodes_train=args.n_episodes_train,
            n_episodes_val=args.n_episodes_val,
            trainable_rff=args.trainable_rff,
        )

    print("\nAll HPO done.")


if __name__ == "__main__":
    main()
