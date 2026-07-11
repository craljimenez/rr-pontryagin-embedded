"""Task 4 — per-image metrics on the PASCAL VOC 2012 test set (725 images).

`metrics_detailed.json`/`metrics_per_class.csv` pool all 725 test images into
one confusion matrix before computing IoU/Dice/etc. This re-runs inference
from the saved `best_model.pth` checkpoints (no retraining) and calls the
SAME `compute_metrics()` used to produce the paper's Table 5 numbers, once
per image instead of once on the pooled batch.

Output: results/seg_pascalvoc_persample.csv
    columns: head, image_idx, iou, dice, precision, recall, pixel_acc
"""
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.seg_pascalvoc import DATASET_ROOT, IMG_SIZE, N_CLASSES, VOID_LABEL
from configs.seg_pascalvoc import RESULTS_DIR as CFG_RESULTS_DIR
from run_seg_pascalvoc import PascalVOCSegDataset, build_unet_model, compute_metrics

RESULTS_DIR = Path(__file__).parent / "results" / "seg_pascalvoc_results"
OUT_CSV = Path(__file__).parent / "results" / "seg_pascalvoc_persample.csv"
HEADS = ["euclidean", "hyperbolic", "pontryagin", "pontryagin_trff"]


def _load_model(head, device):
    # "pontryagin_trff" is model_type="pontryagin" with trainable_rff=True in
    # the checkpoint's saved params (its own results folder, never colliding
    # with the fixed-RFF run) — same handling as generate_pascalvoc_interpretability.
    model_type = "pontryagin" if head == "pontryagin_trff" else head
    ckpt = torch.load(RESULTS_DIR / head / "best_model.pth",
                      map_location=device, weights_only=False)
    params = ckpt.get("params", {})
    if not params:
        p = RESULTS_DIR / head / "hpo" / "best_params.json"
        if p.exists():
            params = json.load(open(p))
    model = build_unet_model(model_type, params)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def per_image_rows(model, dataset, head, device):
    rows = []
    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(1)[0].cpu().reshape(-1)
        label = mask_t.reshape(-1)
        valid = label != VOID_LABEL
        m = compute_metrics(pred[valid], label[valid], N_CLASSES)["global"]
        rows.append({
            "head": head, "image_idx": idx,
            "iou": m["macro_iou"], "dice": m["macro_dice"],
            "precision": m["macro_precision"], "recall": m["macro_recall"],
            "pixel_acc": m["pixel_acc"],
        })
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PascalVOCSegDataset(DATASET_ROOT, "test", IMG_SIZE, augment=False)
    print(f"Test set: {len(dataset)} images")

    all_rows = []
    for head in HEADS:
        print(f"  {head} ...")
        model = _load_model(head, device)
        all_rows += per_image_rows(model, dataset, head, device)
        del model

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}")

    import numpy as np
    for head in HEADS:
        vals = [r["iou"] for r in all_rows if r["head"] == head]
        print(f"  {head:<11s} mIoU = {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
