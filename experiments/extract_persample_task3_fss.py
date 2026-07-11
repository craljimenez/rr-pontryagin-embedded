"""Task 3 — per-episode metrics on the 1-shot few-shot segmentation test set (200 episodes).

`results/fss_sugarcane/test_detailed_1shot.json` stores mean/std/min/max/
median per method but the underlying 200 per-episode values were computed
transiently and discarded (`eval_epoch()` returns only `np.mean(...)`). This
re-runs inference from the saved checkpoints (no retraining) with batch_size=1
so `binary_metrics()` returns a genuine single-episode value per call, over
the same deterministic (seed=42) 200-episode test set used for the paper's
Table 4 numbers.

Output: results/fss_sugarcane_persample.csv
    columns: variant, episode, iou, dice, precision, recall
"""
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.fss_sugarcane import (
    DATASET_ROOT, DEVICE, IMG_SIZE, N_EPISODES_TEST, RESULTS_DIR, TARGET_CLS,
)
from prfe.data.fss_dataset import EpisodicUAVDataset
from run_fss_sugarcane import binary_metrics, build_model

OUT_CSV = Path(__file__).parent / "results" / "fss_sugarcane_persample.csv"

# (results folder name, model_name, trainable_rff) — matches paper Table 4
# plus the tRFF variant used in the interpretability sub-analysis.
VARIANTS = [
    ("euclidean_1shot",           "euclidean",  False),
    ("pontryagin_1shot",          "pontryagin", False),
    ("pontryagin_1shot_trainable","pontryagin", True),
]


def _load_model(run_name, model_name, trainable_rff, device):
    run_dir = RESULTS_DIR / run_name
    params = {}
    hpo_json = run_dir / "hpo" / "best_params.json"
    if hpo_json.exists():
        import json
        params = {k: v for k, v in json.loads(hpo_json.read_text()).items()
                  if not k.startswith("_")}
    model, _ = build_model(model_name, 1, device, params=params,
                           trainable_rff=trainable_rff)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def per_episode_rows(model, loader, run_name):
    rows = []
    for ep, batch in enumerate(loader):
        batch = {k: (v.to(next(model.parameters()).device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        pred = model.predict(batch["support_imgs"], batch["support_masks"], batch["query_img"])
        m = binary_metrics(pred, batch["query_mask"])
        rows.append({"variant": run_name, "episode": ep, **m})
    return rows


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = EpisodicUAVDataset(
        DATASET_ROOT, "test", k_shot=1, n_episodes=N_EPISODES_TEST,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=42,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test episodes: {len(test_ds)}")

    all_rows = []
    for run_name, model_name, trainable_rff in VARIANTS:
        print(f"  {run_name} ...")
        model = _load_model(run_name, model_name, trainable_rff, device)
        all_rows += per_episode_rows(model, loader, run_name)
        del model

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}")

    import numpy as np
    for run_name, _, _ in VARIANTS:
        vals = [r["iou"] for r in all_rows if r["variant"] == run_name]
        print(f"  {run_name:<28s} IoU = {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
