"""Run all paired statistical significance tests for the PRFE paper.

Consumes the per-sample CSVs produced by extract_persample_task{1,2,3,4}_*.py
and applies the test appropriate to each task's outcome type and sample size
(see statistical_tests.py's module docstring for the general rationale):

  Task 1 (UAV segmentation, n=13 images/backbone) — paired Wilcoxon on
    per-image mIoU, PRFE vs. each baseline, within each backbone.
  Task 2 (classification, n=378 samples)          — McNemar on paired
    correct/incorrect outcomes, PRFE-tRFF vs. each baseline.
  Task 3 (few-shot segmentation, n=200 episodes)   — paired Wilcoxon on
    per-episode IoU/Dice/Precision/Recall, PRFE-FSS vs. Euclidean-FSS.
  Task 4 (PASCAL VOC, n=725 images)                — paired Wilcoxon on
    per-image mIoU, PRFE vs. each baseline.

Each task's family of comparisons gets its own Holm-Bonferroni correction
(comparisons across DIFFERENT tasks are not part of the same family — they
answer unrelated questions, so correcting across tasks would be overly
conservative and is not standard practice).

Output: results/statistical_tests_summary.json (all results, machine-readable)
        printed report (human-readable, this is what to read first)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from statistical_tests import mcnemar_test, paired_wilcoxon, run_family

RESULTS_DIR = Path(__file__).parent / "results"
OUT_JSON = RESULTS_DIR / "statistical_tests_summary.json"


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _pivot(rows: list[dict], group_key: str, id_key: str, value_key: str,
          groups: list[str]) -> dict:
    """rows -> {group: {id: value}}, aligned so every group has the same ids
    in the same order (required for pairing)."""
    by_group = defaultdict(dict)
    for r in rows:
        by_group[r[group_key]][r[id_key]] = float(r[value_key])
    ids = sorted(by_group[groups[0]].keys(), key=lambda x: (len(x), x))
    for g in groups:
        assert set(by_group[g].keys()) == set(ids), f"sample-id mismatch for group {g!r}"
    return {g: [by_group[g][i] for i in ids] for g in groups}, ids


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — UAV segmentation (n=13, Wilcoxon per backbone)
# ─────────────────────────────────────────────────────────────────────────────

def task1():
    print("\n" + "#" * 70)
    print("# TASK 1 — UAV sugarcane segmentation (n=13 test images)")
    print("#" * 70)
    rows = _read_csv(RESULTS_DIR / "seg_uav_persample.csv")
    results_all = {}
    for backbone in ("FCN", "UNet"):
        sub = [r for r in rows if r["backbone"] == backbone]
        piv, ids = _pivot(sub, "head", "image_id", "iou", ["euclidean", "hyperbolic", "pontryagin"])
        tests = [
            paired_wilcoxon(piv["pontryagin"], piv["euclidean"],
                            name=f"[{backbone}] PRFE vs Euclidean (mIoU)"),
            paired_wilcoxon(piv["pontryagin"], piv["hyperbolic"],
                            name=f"[{backbone}] PRFE vs Hyperbolic (mIoU)"),
        ]
        payload = run_family(tests, out_path=RESULTS_DIR / f"stats_task1_{backbone.lower()}.json")
        results_all[backbone] = payload
    return results_all


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — classification (n=378, McNemar)
# ─────────────────────────────────────────────────────────────────────────────

def task2():
    print("\n" + "#" * 70)
    print("# TASK 2 — Leaf disease classification (n=378 test samples)")
    print("#" * 70)
    rows = _read_csv(RESULTS_DIR / "cls_sugarcane_persample.csv")
    groups = ["euclidean", "pontryagin", "pontryagin_trff",
              "pontryagin_margin", "pontryagin_margin_trff"]
    piv, ids = _pivot(rows, "variant", "sample_idx", "correct", groups)
    correct = {g: [bool(v) for v in vals] for g, vals in piv.items()}

    tests = [
        mcnemar_test(correct["pontryagin_trff"], correct["euclidean"],
                    name="PRFE-tRFF vs Euclidean"),
        mcnemar_test(correct["pontryagin_trff"], correct["pontryagin"],
                    name="PRFE-tRFF vs PRFE (fixed RFF)"),
        mcnemar_test(correct["pontryagin_trff"], correct["pontryagin_margin"],
                    name="PRFE-tRFF vs P-Margin"),
    ]
    payload = run_family(tests, out_path=RESULTS_DIR / "stats_task2_cls.json")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — few-shot segmentation (n=200 episodes, Wilcoxon x4 metrics)
# ─────────────────────────────────────────────────────────────────────────────

def task3():
    print("\n" + "#" * 70)
    print("# TASK 3 — Few-shot segmentation (n=200 test episodes)")
    print("#" * 70)
    rows = _read_csv(RESULTS_DIR / "fss_sugarcane_persample.csv")
    groups = ["euclidean_1shot", "pontryagin_1shot"]
    tests = []
    for metric in ("iou", "dice", "precision", "recall"):
        piv, ids = _pivot(rows, "variant", "episode", metric, groups)
        tests.append(paired_wilcoxon(
            piv["pontryagin_1shot"], piv["euclidean_1shot"],
            name=f"PRFE-FSS vs Euclidean-FSS ({metric})",
        ))
    payload = run_family(tests, out_path=RESULTS_DIR / "stats_task3_fss.json")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — PASCAL VOC (n=725, Wilcoxon)
# ─────────────────────────────────────────────────────────────────────────────

def task4():
    print("\n" + "#" * 70)
    print("# TASK 4 — PASCAL VOC 2012 (n=725 test images)")
    print("#" * 70)
    rows = _read_csv(RESULTS_DIR / "seg_pascalvoc_persample.csv")
    heads = ["euclidean", "hyperbolic", "pontryagin", "pontryagin_trff"]
    piv, ids = _pivot(rows, "head", "image_idx", "iou", heads)
    tests = [
        paired_wilcoxon(piv["pontryagin"], piv["euclidean"],
                        name="PRFE(fixed) vs Euclidean (mIoU)"),
        paired_wilcoxon(piv["pontryagin"], piv["hyperbolic"],
                        name="PRFE(fixed) vs Hyperbolic (mIoU)"),
        paired_wilcoxon(piv["pontryagin_trff"], piv["euclidean"],
                        name="PRFE-tRFF vs Euclidean (mIoU)"),
        paired_wilcoxon(piv["pontryagin_trff"], piv["hyperbolic"],
                        name="PRFE-tRFF vs Hyperbolic (mIoU)"),
        paired_wilcoxon(piv["pontryagin_trff"], piv["pontryagin"],
                        name="PRFE-tRFF vs PRFE(fixed) (mIoU)"),
    ]
    payload = run_family(tests, out_path=RESULTS_DIR / "stats_task4_voc.json")
    return payload


def main():
    summary = {
        "task1_seg_uav": task1(),
        "task2_cls": task2(),
        "task3_fss": task3(),
        "task4_voc": task4(),
    }
    import json
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}\nAll results saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
