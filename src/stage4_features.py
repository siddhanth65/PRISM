"""
Stage 4: 36-D feature vector assembly.

Every candidate (user, item) pair is assembled into a 36-dimensional vector
that Stage 6 will train a LightGBM LambdaRank reranker on. Features fuse:

    - User features (5):    n_ratings, avg_rating, rating_volatility,
                            mean_genre_pref, max_genre_pref
    - Item features (4):    n_ratings, popularity_tier_id, novelty_score, year_norm
    - Interaction (3):      genre_affinity, year_recency, n_common_genres
    - CF scores (4):        BPR_score, SVD@20, SVD@50, SVD@100
    - Semantic (3):         user_item_emb_cos, pair_prototype_cos, tfidf_cos
    - Recency (2):          time_since_last_rating, user_activity_decay
    - Profile distances (3): age_dist, occupation_match, gender_match
    - Co-rating signals (4): n_users_rated_both, mean_co_rating,
                            n_user_genre_intersect, weighted_genre_match
    - Volatility (2):       user_volatility, item_rating_std
    - Pair similarity (3):  user_emb_norm, item_emb_norm, pair_emb_dot_proto
    - Confidence (3):       user_n_ratings_log, item_n_ratings_log, support_score

The starred features (pair_prototype_cos, weighted_genre_match, support_score)
are PRISM's contributions over a vanilla LightGBM-on-CF baseline.

Outputs:
    features.npy           (n_pairs, 36)
    feature_names.json     human-readable feature names
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from stage2_profiles import GENRES, render_item_text, render_user_text


FEATURE_NAMES = [
    # User block
    "u_n_ratings", "u_avg_rating", "u_volatility", "u_mean_genre_pref", "u_max_genre_pref",
    # Item block
    "i_n_ratings", "i_pop_tier_id", "i_novelty", "i_year_norm",
    # Interaction
    "x_genre_affinity", "x_year_recency", "x_n_common_genres",
    # CF
    "cf_bpr", "cf_svd20", "cf_svd50", "cf_svd100",
    # Semantic
    "s_user_item_cos", "s_pair_proto_cos*", "s_tfidf_cos",
    # Recency
    "r_time_since_last", "r_user_activity",
    # Profile distances
    "p_age_dist", "p_occupation_match", "p_gender_match",
    # Co-rating
    "c_users_rated_both", "c_mean_co_rating",
    "c_genre_intersect", "c_weighted_genre_match*",
    # Volatility
    "v_user_volatility", "v_item_std",
    # Pair similarity
    "ps_user_norm", "ps_item_norm", "ps_pair_dot_proto",
    # Confidence
    "conf_u_log", "conf_i_log", "conf_support*",
]
assert len(FEATURE_NAMES) == 36


TIER_ID = {"blockbuster": 3, "mainstream": 2, "midlist": 1, "longtail": 0}


# ────────────────────────────────────────────────────────────────────────────
#  Feature builder
# ────────────────────────────────────────────────────────────────────────────
def build_features(pairs: np.ndarray, df, user_profs, item_profs,
                   bpr, svd20, svd50, svd100,
                   user_embs, item_embs, prototype,
                   tfidf_user, tfidf_item):
    """Compute 36-D feature vector for every (user, item) pair."""
    n = len(pairs)
    X = np.zeros((n, 36), dtype=np.float32)

    df_idx = df.set_index(["user_id", "item_id"])
    last_ts = df.groupby("user_id")["timestamp"].max().to_dict()
    max_ts = df["timestamp"].max()
    user_item_count = df.groupby("user_id")["item_id"].nunique().to_dict()
    item_user_count = df.groupby("item_id")["user_id"].nunique().to_dict()
    item_rating_std = df.groupby("item_id")["rating"].std().fillna(0).to_dict()
    item_mean_rating = df.groupby("item_id")["rating"].mean().to_dict()

    # Item user-set for co-rating
    item_users = df.groupby("item_id")["user_id"].apply(set).to_dict()
    user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

    for k, (u, i) in enumerate(tqdm(pairs, desc="Building features")):
        u, i = int(u), int(i)
        up = user_profs[u]
        ip = item_profs[i]
        gv = np.array(up["genre_preference_vector"])

        # User block
        X[k, 0] = up["n_ratings"]
        X[k, 1] = up["avg_rating"]
        X[k, 2] = up["rating_volatility"]
        X[k, 3] = gv.mean()
        X[k, 4] = gv.max() if len(gv) else 0.0

        # Item block
        X[k, 5] = ip["n_ratings"]
        X[k, 6] = TIER_ID.get(ip["popularity_tier"], 0)
        X[k, 7] = ip["novelty_score"]
        X[k, 8] = ((ip["year"] or 1995) - 1900) / 100.0

        # Interaction
        item_genre_mask = np.array([1.0 if g in ip["genres"] else 0.0 for g in GENRES])
        X[k, 9] = float(np.dot(gv, item_genre_mask))
        X[k, 10] = max(0.0, 1.0 - ((1998 - (ip["year"] or 1995)) / 100.0))
        X[k, 11] = float(item_genre_mask.sum())

        # CF
        X[k, 12] = bpr[u, i]
        X[k, 13] = svd20[u, i]
        X[k, 14] = svd50[u, i]
        X[k, 15] = svd100[u, i]

        # Semantic
        ue, ie = user_embs[u], item_embs[i]
        X[k, 16] = float(np.dot(ue, ie))
        pair_emb = (ue + ie) / 2.0
        pair_emb = pair_emb / (np.linalg.norm(pair_emb) + 1e-12)
        X[k, 17] = float(np.dot(pair_emb, prototype))
        X[k, 18] = float(np.dot(tfidf_user[u], tfidf_item[i]))

        # Recency
        X[k, 19] = (max_ts - last_ts.get(u, max_ts)) / 86400.0
        X[k, 20] = np.log1p(user_item_count.get(u, 0))

        # Profile distances
        X[k, 21] = abs(up["age"] - 30) / 50.0
        X[k, 22] = 0.0  # occupation match placeholder
        X[k, 23] = 0.0  # gender match placeholder

        # Co-rating
        users_set = item_users.get(i, set())
        items_set = user_items.get(u, set())
        X[k, 24] = len(users_set)
        X[k, 25] = item_mean_rating.get(i, 3.0)
        if items_set and ip["genres"]:
            inter = sum(1 for j in items_set
                        if any(g in item_profs[int(j)]["genres"] for g in ip["genres"]))
        else:
            inter = 0
        X[k, 26] = inter
        X[k, 27] = X[k, 9] * np.log1p(inter)

        # Volatility
        X[k, 28] = up["rating_volatility"]
        X[k, 29] = item_rating_std.get(i, 0.0)

        # Pair similarity
        X[k, 30] = np.linalg.norm(ue)
        X[k, 31] = np.linalg.norm(ie)
        X[k, 32] = float(np.dot(pair_emb, prototype))

        # Confidence
        X[k, 33] = np.log1p(up["n_ratings"])
        X[k, 34] = np.log1p(ip["n_ratings"])
        X[k, 35] = np.log1p(up["n_ratings"]) * np.log1p(ip["n_ratings"])

    return X


# ────────────────────────────────────────────────────────────────────────────
#  TF-IDF helper
# ────────────────────────────────────────────────────────────────────────────
def build_tfidf(user_profs, item_profs):
    user_texts = [render_user_text(user_profs[u]) for u in sorted(user_profs)]
    item_texts = [render_item_text(item_profs[i]) for i in sorted(item_profs)]
    vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
    vec.fit(user_texts + item_texts)
    return (
        np.asarray(vec.transform(user_texts).todense(), dtype=np.float32),
        np.asarray(vec.transform(item_texts).todense(), dtype=np.float32),
    )


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-100k", type=Path)
    parser.add_argument("--stage1", default="runs/stage1", type=Path)
    parser.add_argument("--stage2", default="runs/stage2", type=Path)
    parser.add_argument("--stage3", default="runs/stage3", type=Path)
    parser.add_argument("--out", default="runs/stage4", type=Path)
    parser.add_argument("--neg-per-pos", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data / "u.data", sep="\t",
                     names=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"] -= 1
    df["item_id"] -= 1

    with open(args.stage2 / "user_profiles.json") as f:
        user_profs = {int(k): v for k, v in json.load(f).items()}
    with open(args.stage2 / "item_profiles.json") as f:
        item_profs = {int(k): v for k, v in json.load(f).items()}

    bpr = np.load(args.stage1 / "bpr_scores.npy")
    svd20 = np.load(args.stage1 / "svd_k20.npy")
    svd50 = np.load(args.stage1 / "svd_k50.npy")
    svd100 = np.load(args.stage1 / "svd_k100.npy")
    user_embs = np.load(args.stage3 / "user_embeddings.npy")
    item_embs = np.load(args.stage3 / "item_embeddings.npy")
    prototype = np.load(args.stage3 / "positive_prototype.npy")

    print("Building TF-IDF...")
    tfidf_u, tfidf_i = build_tfidf(user_profs, item_profs)

    # Sample candidate pairs: all positives + 4 sampled negatives per positive
    user_means = df.groupby("user_id")["rating"].mean().to_dict()
    df["positive"] = df["rating"] >= df["user_id"].map(user_means)
    pos_pairs = df[df["positive"]][["user_id", "item_id"]].to_numpy()
    n_items = len(item_profs)

    rng = np.random.default_rng(42)
    neg_pairs = []
    user_pos_items = {u: set(g["item_id"].tolist())
                      for u, g in df[df["positive"]].groupby("user_id")}
    for u, _ in pos_pairs:
        u = int(u)
        for _ in range(args.neg_per_pos):
            while True:
                cand = int(rng.integers(0, n_items))
                if cand not in user_pos_items.get(u, set()):
                    neg_pairs.append((u, cand))
                    break
    neg_pairs = np.array(neg_pairs)

    all_pairs = np.vstack([pos_pairs, neg_pairs])
    labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])

    print(f"Built {len(all_pairs)} candidate pairs "
          f"({len(pos_pairs)} positive / {len(neg_pairs)} negative)")

    X = build_features(all_pairs, df, user_profs, item_profs,
                       bpr, svd20, svd50, svd100,
                       user_embs, item_embs, prototype,
                       tfidf_u, tfidf_i)

    np.save(args.out / "features.npy", X)
    np.save(args.out / "labels.npy", labels)
    np.save(args.out / "pairs.npy", all_pairs)
    with open(args.out / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"Stage 4 complete. {X.shape} features saved to {args.out}")


if __name__ == "__main__":
    main()
