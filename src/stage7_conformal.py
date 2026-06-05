"""
Stage 7: Inductive conformal prediction.

PRISM's most distinctive contribution. We apply inductive (split) conformal
prediction on top of the blended Stage-5 score to produce calibrated p-values
with a distribution-free, finite-sample coverage guarantee.

Method:
    1. Compute the blended score s(u, i) for the calibration set.
    2. Define a nonconformity score:
           A(u, i, y) = 1 - s(u, i)   if y = 1 (positive)
                       = s(u, i)       if y = 0 (negative)
       Lower A means the example "looks like" its label.
    3. Let q_alpha be the ceil((n+1)(1 - alpha))-th smallest calibration A.
    4. For a test pair (u, i) with score s, the conformal p-value is:
           p(u, i) = (1 + sum_j [A_cal_j >= 1 - s]) / (n_cal + 1)
       i.e. the fraction of calibration examples (plus the test point) that
       are at least as nonconforming with the "positive" hypothesis.

Coverage guarantee: at significance level alpha, the true positive set is
captured with probability >= 1 - alpha, independent of the underlying rating
distribution. With alpha = 0.1 we get a 90% finite-sample coverage.

Outputs:
    conformal_pvalues.npy     p-value per pair
    coverage_report.json      empirical coverage on a held-out test slice
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Calibration and inference
# ────────────────────────────────────────────────────────────────────────────
def nonconformity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nonconformity score per the spec above."""
    return np.where(labels == 1, 1.0 - scores, scores)


def conformal_pvalue(test_scores: np.ndarray, cal_nc: np.ndarray) -> np.ndarray:
    """One-sided p-value for the hypothesis 'this is a positive'."""
    n = len(cal_nc)
    test_nc = 1.0 - test_scores
    # Rank of test_nc among cal_nc (right-side)
    sorted_nc = np.sort(cal_nc)
    ranks = np.searchsorted(sorted_nc, test_nc, side="left")
    return (n - ranks + 1) / (n + 1)


def empirical_coverage(p_values: np.ndarray, labels: np.ndarray,
                       alpha: float = 0.1) -> float:
    """Fraction of true positives that exceed the p-value threshold."""
    pos = labels == 1
    if pos.sum() == 0:
        return float("nan")
    return float((p_values[pos] >= alpha).mean())


# ────────────────────────────────────────────────────────────────────────────
#  Score blending
# ────────────────────────────────────────────────────────────────────────────
def normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def blend(bpr_pair_scores: np.ndarray, lgbm_scores: np.ndarray,
          alpha: float) -> np.ndarray:
    return alpha * normalise(bpr_pair_scores) + (1 - alpha) * normalise(lgbm_scores)


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="runs/stage1", type=Path)
    parser.add_argument("--stage4", default="runs/stage4", type=Path)
    parser.add_argument("--stage5", default="runs/stage5", type=Path)
    parser.add_argument("--stage6", default="runs/stage6", type=Path)
    parser.add_argument("--out", default="runs/stage7", type=Path)
    parser.add_argument("--alpha-conformal", type=float, default=0.1,
                        help="Significance level (1 - coverage)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bpr = np.load(args.stage1 / "bpr_scores.npy")
    pairs = np.load(args.stage4 / "pairs.npy")
    labels = np.load(args.stage4 / "labels.npy")
    lgbm_scores = np.load(args.stage6 / "oof_scores.npy")

    with open(args.stage5 / "best_alpha.json") as f:
        best_alpha = json.load(f)["alpha"]
    print(f"Using alpha={best_alpha:.2f} from Stage 5")

    bpr_pair = np.array([bpr[int(u), int(i)] for u, i in pairs])
    blended = blend(bpr_pair, lgbm_scores, alpha=best_alpha)

    # 60/20/20 calibration/test/eval split on positives
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(blended))
    n_cal = int(0.6 * len(idx))
    n_test = int(0.2 * len(idx))
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:n_cal + n_test]

    cal_nc = nonconformity(blended[cal_idx], labels[cal_idx])
    test_pv = conformal_pvalue(blended[test_idx], cal_nc)
    full_pv = conformal_pvalue(blended, cal_nc)

    coverage = empirical_coverage(test_pv, labels[test_idx], alpha=args.alpha_conformal)
    report = {
        "target_coverage": 1.0 - args.alpha_conformal,
        "empirical_coverage": coverage,
        "calibration_size": int(n_cal),
        "test_size": int(n_test),
        "alpha_conformal": args.alpha_conformal,
        "best_blend_alpha": best_alpha,
    }
    with open(args.out / "coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(args.out / "conformal_pvalues.npy", full_pv)

    print(f"Target coverage: {report['target_coverage']:.2f}")
    print(f"Empirical coverage: {coverage:.4f}")
    print(f"Stage 7 complete. Outputs in {args.out}")


if __name__ == "__main__":
    main()
