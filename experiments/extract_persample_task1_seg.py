"""Task 1 — per-image metrics on the UAV sugarcane test set (13 images).

The saved results (`results/seg_uav*/<head>/metrics_detailed.json`) pool all
13 test images into one confusion matrix before computing IoU/Dice/etc., so
no per-image breakdown exists on disk — this script re-runs inference from
the already-trained `best_model.pth` checkpoints (no retraining) and calls
the SAME `compute_metrics()` used to produce the paper's Table 1 numbers,
once per image instead of once on the pooled batch, so per-image values are
directly comparable to (and average out to) the published aggregates.

Output: results/seg_uav_persample.csv
    columns: backbone, head, image_id, iou, dice, precision, recall, pixel_acc
"""
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.seg_uav import (
    AVAILABLE_MODELS as FCN_HEADS, BINARY_ONLY_SUGARCANE, BINARY_SUGARCANE,
    DATASET_ROOT, IMG_SIZE, N_CLASSES,
)
from configs.seg_uav import RESULTS_DIR as FCN_RESULTS_DIR
from configs.seg_uav_unet import RESULTS_DIR as UNET_RESULTS_DIR
from run_seg_uav import UAVSegDataset, build_model as fcn_build_model, compute_metrics
from run_seg_uav_unet import build_unet_model

OUT_CSV = Path(__file__).parent / "results" / "seg_uav_persample.csv"
HEADS = ["euclidean", "hyperbolic", "pontryagin"]   # matches paper Table 1 (excludes vanilla)


def _load_params(results_dir: Path, head: str) -> dict:
    p = results_dir / head / "hpo" / "best_params.json"
    return json.load(open(p)) if p.exists() else {}


def _load_model(build_fn, results_dir: Path, head: str, device):
    params = _load_params(results_dir, head)
    model = build_fn(head, params)
    ckpt = torch.load(results_dir / head / "best_model.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def per_image_rows(model, dataset, backbone_name, head, device):
    rows = []
    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        stem = dataset.samples[idx]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(1)[0].cpu().reshape(-1)
        label = mask_t.reshape(-1)
        m = compute_metrics(pred, label, N_CLASSES)["global"]
        rows.append({
            "backbone": backbone_name, "head": head, "image_id": stem,
            "iou": m["macro_iou"], "dice": m["macro_dice"],
            "precision": m["macro_precision"], "recall": m["macro_recall"],
            "pixel_acc": m["pixel_acc"],
        })
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = UAVSegDataset(
        root=DATASET_ROOT, split="test", img_size=IMG_SIZE, augment=False,
        binary_sugarcane=BINARY_SUGARCANE, only_sugarcane=BINARY_ONLY_SUGARCANE,
    )
    print(f"Test set: {len(dataset)} images")

    all_rows = []
    for head in HEADS:
        print(f"  FCN  / {head} ...")
        model = _load_model(fcn_build_model, FCN_RESULTS_DIR, head, device)
        all_rows += per_image_rows(model, dataset, "FCN", head, device)
        del model

        print(f"  UNet / {head} ...")
        model = _load_model(build_unet_model, UNET_RESULTS_DIR, head, device)
        all_rows += per_image_rows(model, dataset, "UNet", head, device)
        del model

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}")

    # sanity check against the published aggregate: mean over 13 images should
    # match Table 1's mean±std for each backbone/head combination
    import numpy as np
    for backbone in ("FCN", "UNet"):
        for head in HEADS:
            vals = [r["iou"] for r in all_rows if r["backbone"] == backbone and r["head"] == head]
            print(f"  {backbone:4s} {head:10s} mIoU = {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
