"""Task 2 — per-sample predictions on the Sugarcane Leaf Disease test set (378 samples).

`metrics_test.json` only stores aggregated accuracy/F1/confusion-matrix; no
per-sample predicted-vs-true record exists on disk. This re-runs inference
from the saved `best_model.pth` checkpoints (no retraining) over the same
deterministic test split (`SEED`-derived, DataLoader shuffle=False) used to
produce the paper's Table 2 numbers.

Output: results/cls_sugarcane_persample.csv
    columns: variant, sample_idx, true_label, pred_label, correct
"""
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.cls_sugarcane import DEVICE, RESULTS_DIR
from run_cls_sugarcane import (
    _find_dataset_root, build_cls_model, build_dataloaders, download_dataset,
)

OUT_CSV = Path(__file__).parent / "results" / "cls_sugarcane_persample.csv"

# (results folder name, model_type, trainable_rff) — matches paper Table 2 rows
VARIANTS = [
    ("euclidean",             "euclidean",         False),
    ("pontryagin",            "pontryagin",         False),
    ("pontryagin_trff",       "pontryagin",         True),
    ("pontryagin_margin",     "pontryagin_margin",  False),
    ("pontryagin_margin_trff","pontryagin_margin",  True),
]


def _load_model(run_name, model_type, trainable_rff, device):
    ckpt_path = RESULTS_DIR / run_name / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    params = ckpt.get("params", {})
    tr = ckpt.get("trainable_rff", trainable_rff)
    model = build_cls_model(model_type, n_classes=len(classes), params=params,
                            trainable_rff=tr)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), classes


@torch.no_grad()
def per_sample_rows(model, loader, run_name, device):
    rows = []
    idx = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        for p, y in zip(preds.tolist(), labels.tolist()):
            rows.append({
                "variant": run_name, "sample_idx": idx,
                "true_label": y, "pred_label": p, "correct": int(p == y),
            })
            idx += 1
    return rows


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_root = os.environ.get("SUGARCANE_DATA_ROOT")
    data_root = _find_dataset_root(Path(env_root)) if env_root else _find_dataset_root(download_dataset())
    data = build_dataloaders(data_root)
    print(f"Test set: {len(data['test_ds'])} samples, classes={data['classes']}")

    all_rows = []
    for run_name, model_type, trainable_rff in VARIANTS:
        print(f"  {run_name} ...")
        model, classes = _load_model(run_name, model_type, trainable_rff, device)
        assert classes == data["classes"], f"class order mismatch for {run_name}"
        all_rows += per_sample_rows(model, data["test_dl"], run_name, device)
        del model

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}")

    import numpy as np
    for run_name, _, _ in VARIANTS:
        acc = np.mean([r["correct"] for r in all_rows if r["variant"] == run_name])
        n = sum(1 for r in all_rows if r["variant"] == run_name)
        print(f"  {run_name:<24s} acc = {acc:.4f}  (n={n})")


if __name__ == "__main__":
    main()
