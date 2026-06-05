"""
Stage 5: Evaluation and metric-guided tuning.

Implements Hit Rate at K and Normalised Discounted Cumulative Gain at K, the
two complementary ranking metrics PRISM is evaluated on, and grids the blend
weight alpha to choose the BPR/LightGBM combination that maximises NDCG@10 on
a held-out validation slice.

Why both metrics:
    HR@K is recall-flavoured: did at least one relevant item make the top-K?
    NDCG@K is position-sensitive: items ranked higher score more, so a model
    that buries good items is punished. PRISM optimises NDCG@K directly via
    LambdaRank in Stage 6, then this stage tunes the blend with LightGBM.

The final ranking score is:
    score(u, i) = alpha * BPR_norm(u, i) + (1 - alpha) * LightGBM(features)

Output:
    alpha_results.csv    grid search results
    best_alpha.json      selected alpha and its NDCG@10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
#  Ranking metrics
# ────────────────────────────────────────────────────────────────────────────
def hit_rate_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Fraction of users with at least one relevant item in the top-K."""
    order = np.argsort(-scores)[:k]
    return float(labels[order].sum() > 0)


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Normalised DCG. DCG = sum_i  rel_i / log2(i + 2).

    IDCG is computed from the labels sorted descending, which is the perfect
    ranking. NDCG is in [0, 1] and 1.0 only if every relevant item is ranked
    above every irrelevant one.
    """
    order = np.argsort(-scores)[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = (gains * discounts).sum()

    ideal_gains = np.sort(labels)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = (ideal_gains * ideal_discounts).sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


def per_user_metrics(scores_dict, labels_dict, k=10):
    """Average HR@K and NDCG@K across users."""
    hrs, ndcgs = [], []
    for u in scores_dict:
        if u not in labels_dict:
            continue
        hrs.append(hit_rate_at_k(scores_dict[u], labels_dict[u], k))
        ndcgs.append(ndcg_at_k(scores_dict[u], labels_dict[u], k))
    return float(np.mean(hrs)), float(np.mean(ndcgs))


# ────────────────────────────────────────────────────────────────────────────
#  Alpha grid search
# ────────────────────────────────────────────────────────────────────────────
def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalisation, robust to the constant-score edge case."""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-12)


def grid_search_alpha(val_users, bpr_scores, lgbm_scores, val_pairs, val_labels,
                      alphas=None, k=10) -> pd.DataFrame:
    if alphas is None:
        alphas = np.arange(0.0, 0.81, 0.1)

    rows = []
    for alpha in alphas:
        per_user_scores = {}
        per_user_labels = {}
        for u in val_users:
            mask = val_pairs[:, 0] == u
            if mask.sum() == 0:
                continue
            items_u = val_pairs[mask, 1]
            bpr_u = normalise_scores(bpr_scores[u, items_u])
            lgbm_u = normalise_scores(lgbm_scores[mask])
            blend = alpha * bpr_u + (1 - alpha) * lgbm_u
            per_user_scores[int(u)] = blend
            per_user_labels[int(u)] = val_labels[mask]
        hr, ndcg = per_user_metrics(per_user_scores, per_user_labels, k=k)
        rows.append({"alpha": float(alpha), "hr@10": hr, "ndcg@10": ndcg})
        print(f"  alpha={alpha:.2f} -> HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="runs/stage1", type=Path)
    parser.add_argument("--stage4", default="runs/stage4", type=Path)
    parser.add_argument("--lgbm-scores", default="runs/stage6/oof_scores.npy", type=Path)
    parser.add_argument("--out", default="runs/stage5", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bpr = np.load(args.stage1 / "bpr_scores.npy")
    pairs = np.load(args.stage4 / "pairs.npy")
    labels = np.load(args.stage4 / "labels.npy")
    lgbm_scores = np.load(args.lgbm_scores)

    rng = np.random.default_rng(42)
    all_users = np.unique(pairs[:, 0])
    val_users = rng.choice(all_users, size=int(0.15 * len(all_users)), replace=False)
    print(f"Held-out validation: {len(val_users)} users")

    print("Grid searching alpha...")
    results = grid_search_alpha(val_users, bpr, lgbm_scores, pairs, labels)
    results.to_csv(args.out / "alpha_results.csv", index=False)

    best = results.iloc[results["ndcg@10"].idxmax()]
    with open(args.out / "best_alpha.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    print(f"\nBest alpha={best['alpha']:.2f}, NDCG@10={best['ndcg@10']:.4f}")
    print(f"Stage 5 complete. Results in {args.out}")


if __name__ == "__main__":
    main()
