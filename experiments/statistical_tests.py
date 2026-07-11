"""Statistical significance testing for paired model comparisons.

All four PRFE paper tasks compare 2+ heads (Euclidean / Hyperbolic / PRFE)
evaluated on the SAME test set (same images, same episodes, same samples),
so every comparison here is a *paired* comparison — never an unpaired/
independent-samples test, which would throw away the pairing information
and be statistically weaker and technically wrong for this design.

Two paired test families are implemented, chosen by outcome type:

  * Continuous per-sample metric (IoU, Dice, ...) for two heads on the same
    n samples  ->  Wilcoxon signed-rank test (paired_wilcoxon). Preferred
    over a paired t-test here because IoU/Dice are bounded in [0, 1] and
    frequently skewed (mass near 0 or 1), so the t-test's normality
    assumption is questionable — this holds regardless of n, but matters
    most exactly where n is small (Task 1, n=13) and the CLT can't be
    invoked to rescue non-normality.

  * Paired binary outcome (correct/incorrect per sample) for two
    classifiers on the same n samples  ->  McNemar's test (mcnemar_test).
    This is the standard test for this exact design (Dietterich, 1998,
    "Approximate Statistical Tests for Comparing Supervised Classification
    Learning Algorithms", Neural Computation) — it conditions on the
    *discordant* pairs only (cases where the two classifiers disagree),
    which is what actually carries information about which classifier is
    better; it deliberately ignores pairs where both are right or both are
    wrong, since those are uninformative about *which* model is better.

For every test this module also reports:
  * an effect size (matched-pairs rank-biserial correlation for Wilcoxon;
    the discordant-pair odds ratio for McNemar) — a p-value alone doesn't
    say whether a difference is practically meaningful, especially with
    n in the hundreds where tiny, unimportant differences become
    "significant";
  * a bootstrap percentile confidence interval for the paired difference
    (continuous case) — useful as a sanity check independent of the
    Wilcoxon asymptotics, and the primary evidence at very small n (Task 1,
    n=13) where any single test's asymptotic guarantees are weak and a
    resampling-based interval is more informative than a point p-value.

When several comparisons are run within the same task/table (e.g. PRFE vs.
Euclidean AND PRFE vs. Hyperbolic on the same 13 images), holm_bonferroni
must be applied across that family before quoting significance — running
k tests and reporting whichever come out under 0.05 uncorrected is exactly
the multiple-comparisons error this exists to prevent.

Usage:
    from statistical_tests import paired_wilcoxon, mcnemar_test, holm_bonferroni

    r = paired_wilcoxon(prfe_iou, euclidean_iou, name="PRFE vs Euclidean")
    print(r.summary())

    r2 = mcnemar_test(prfe_correct, euclidean_correct, name="PRFE vs Euclidean")
    print(r2.summary())

    pvals = [r.p_value, r2.p_value]
    adj = holm_bonferroni(pvals)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PairedTestResult:
    name: str
    test: str
    n: int
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    ci_method: str
    notes: str = ""

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 \
            else "*" if self.p_value < 0.05 else "ns"
        return (
            f"{self.name}  [{self.test}, n={self.n}]\n"
            f"  mean diff   = {self.mean_diff:+.4f}  (median {self.median_diff:+.4f})\n"
            f"  {self.ci_method} 95% CI = [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"  statistic   = {self.statistic:.4f}\n"
            f"  p-value     = {self.p_value:.4g}  ({sig})\n"
            f"  effect size = {self.effect_size_name} = {self.effect_size:.3f}"
            + (f"\n  note: {self.notes}" if self.notes else "")
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class McNemarResult:
    name: str
    n: int
    n_a_only: int   # a correct, b wrong
    n_b_only: int   # b correct, a wrong
    n_discordant: int
    statistic: float
    p_value: float
    method: str
    odds_ratio: float
    notes: str = ""

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 \
            else "*" if self.p_value < 0.05 else "ns"
        return (
            f"{self.name}  [McNemar {self.method}, n={self.n}]\n"
            f"  discordant pairs = {self.n_discordant} "
            f"(a-only={self.n_a_only}, b-only={self.n_b_only})\n"
            f"  statistic        = {self.statistic:.4f}\n"
            f"  p-value          = {self.p_value:.4g}  ({sig})\n"
            f"  odds ratio (a:b) = {self.odds_ratio:.3f}"
            + (f"\n  note: {self.notes}" if self.notes else "")
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI for a paired difference
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci_paired(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference mean(a - b).

    Resamples PAIRS (not a and b independently) with replacement, preserving
    the pairing structure — resampling a and b separately would be wrong
    here, since it would destroy the within-sample correlation the paired
    design relies on.
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank (paired, continuous metric)
# ─────────────────────────────────────────────────────────────────────────────

def paired_wilcoxon(
    a: Sequence[float], b: Sequence[float], name: str = "a vs b",
    alternative: str = "two-sided", n_boot: int = 10_000, seed: int = 42,
) -> PairedTestResult:
    """Paired Wilcoxon signed-rank test on a - b, plus bootstrap CI.

    `method='auto'` (scipy default) uses the exact permutation distribution
    for small, tie-free samples (relevant for Task 1, n=13) and the normal
    approximation otherwise (Task 3 n=200, Task 4 n=725) — the right choice
    is made automatically based on n and whether ties/zeros are present.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match in length: {a.shape} vs {b.shape}")
    n = len(a)
    diff = a - b

    notes = ""
    if np.allclose(diff, 0):
        # Degenerate case: identical predictions on every sample.
        return PairedTestResult(
            name=name, test="Wilcoxon signed-rank", n=n, statistic=0.0,
            p_value=1.0, effect_size_name="rank-biserial r", effect_size=0.0,
            mean_diff=0.0, median_diff=0.0, ci_low=0.0, ci_high=0.0,
            ci_method="bootstrap percentile",
            notes="all paired differences are exactly zero",
        )

    n_zero = int(np.sum(diff == 0))
    if n_zero > 0:
        notes = (f"{n_zero}/{n} pairs had zero difference and were dropped "
                  f"(zero_method='wilcox', scipy default)")

    res = stats.wilcoxon(diff, alternative=alternative, method="auto",
                         zero_method="wilcox")
    n_eff = n - n_zero  # effective n after dropping zero-differences
    # Matched-pairs rank-biserial correlation: r = 2*(W_max_possible - W)/W_max_possible
    # where W is the smaller of the two rank sums; equivalently derived from
    # the Wilcoxon statistic and n_eff (King & Minium, 2003).
    w_max = n_eff * (n_eff + 1) / 2
    r_rb = 1 - (2 * res.statistic) / w_max if w_max > 0 else 0.0

    ci_low, ci_high = _bootstrap_ci_paired(a, b, n_boot=n_boot, seed=seed)

    return PairedTestResult(
        name=name, test="Wilcoxon signed-rank", n=n,
        statistic=float(res.statistic), p_value=float(res.pvalue),
        effect_size_name="matched-pairs rank-biserial r", effect_size=float(r_rb),
        mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
        ci_low=ci_low, ci_high=ci_high, ci_method="bootstrap percentile (10k)",
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's test (paired, binary correct/incorrect)
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    correct_a: Sequence[bool], correct_b: Sequence[bool], name: str = "a vs b",
) -> McNemarResult:
    """Exact McNemar test on paired correct/incorrect outcomes.

    Uses the exact binomial form (Dietterich 1998): among the n_discordant
    samples where the two classifiers disagree, test whether the split is
    50/50 under the null (classifiers equally likely to be the one that's
    right when they disagree) via an exact two-sided binomial test. The
    exact binomial form is preferred over the chi-square approximation
    whenever n_discordant is not large (<~25 is the traditional cutoff,
    but scipy's binomtest is exact regardless, so there's no reason not to
    use it even for larger discordant counts).
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Paired arrays must match in length")
    n = len(correct_a)

    a_only = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    b_only = int(np.sum(~correct_a & correct_b))   # b right, a wrong
    n_disc = a_only + b_only

    if n_disc == 0:
        return McNemarResult(
            name=name, n=n, n_a_only=0, n_b_only=0, n_discordant=0,
            statistic=0.0, p_value=1.0, method="exact binomial",
            odds_ratio=1.0, notes="no discordant pairs — models agree on every sample",
        )

    binom = stats.binomtest(a_only, n_disc, p=0.5, alternative="two-sided")
    odds_ratio = a_only / b_only if b_only > 0 else float("inf")

    return McNemarResult(
        name=name, n=n, n_a_only=a_only, n_b_only=b_only, n_discordant=n_disc,
        statistic=float(a_only), p_value=float(binom.pvalue),
        method="exact binomial", odds_ratio=float(odds_ratio),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-comparison correction
# ─────────────────────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction for a family of p-values.

    Chosen over plain Bonferroni because it is uniformly more powerful
    (rejects at least as many hypotheses) while controlling the same
    family-wise error rate; chosen over Benjamini-Hochberg (FDR) because
    with only a handful of comparisons per task here, controlling the
    probability of ANY false positive (FWER) is the more conservative and
    appropriate standard for confirmatory claims in a paper table, rather
    than the expected proportion of false positives (FDR), which is more
    suited to large-scale screening.

    Returns a dict with the sorted order, Holm-adjusted p-values (in the
    ORIGINAL input order), and a boolean reject array at `alpha`.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    reject = adjusted < alpha
    return {
        "p_raw": p.tolist(),
        "p_holm": adjusted.tolist(),
        "reject_at_alpha": reject.tolist(),
        "alpha": alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run + print + save a family of tests together
# ─────────────────────────────────────────────────────────────────────────────

def run_family(
    results: list[PairedTestResult | McNemarResult], out_path: Path | None = None,
    alpha: float = 0.05,
) -> dict:
    """Apply Holm-Bonferroni across `results` (a family of related tests run
    on the same task/table), print a summary, and optionally save JSON."""
    pvals = [r.p_value for r in results]
    holm = holm_bonferroni(pvals, alpha=alpha)

    print(f"\n{'='*70}")
    for r, p_holm, rej in zip(results, holm["p_holm"], holm["reject_at_alpha"]):
        print(r.summary())
        print(f"  p (Holm-corrected, family of {len(results)}) = {p_holm:.4g}"
              f"  -> {'reject H0' if rej else 'fail to reject H0'} at alpha={alpha}")
        print("-" * 70)

    payload = {
        "alpha": alpha,
        "family_size": len(results),
        "tests": [
            {**r.to_dict(), "p_holm": p_holm, "reject_at_alpha": bool(rej)}
            for r, p_holm, rej in zip(results, holm["p_holm"], holm["reject_at_alpha"])
        ],
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved -> {out_path}")
    return payload


if __name__ == "__main__":
    # Self-test with synthetic data (no repo dependencies) — sanity check
    # that both tests behave as expected before wiring in real results.
    rng = np.random.default_rng(0)

    print("### Sanity check: paired_wilcoxon on synthetic IoU-like data ###")
    n = 13
    euclidean = np.clip(rng.normal(0.75, 0.15, n), 0, 1)
    prfe      = np.clip(euclidean + rng.normal(0.08, 0.05, n), 0, 1)  # PRFE consistently a bit higher
    r1 = paired_wilcoxon(prfe, euclidean, name="PRFE vs Euclidean (synthetic, n=13)")
    print(r1.summary())

    print("\n### Sanity check: mcnemar_test on synthetic correct/incorrect ###")
    n2 = 378
    correct_euc  = rng.random(n2) < 0.85
    correct_prfe = rng.random(n2) < 0.90
    r2 = mcnemar_test(correct_prfe, correct_euc, name="PRFE-tRFF vs Euclidean (synthetic, n=378)")
    print(r2.summary())

    print("\n### Sanity check: holm_bonferroni ###")
    print(holm_bonferroni([0.001, 0.03, 0.04, 0.2]))
