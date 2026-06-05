"""
Stage 2: User and item JSON profile builder.

Produces structured text profiles for every user and item. These profiles are
the surface on which Stage 3 runs MiniLM, so the quality of the language here
matters: a profile is essentially a sentence that captures who a user is or
what a movie is, in a form a sentence encoder can usefully embed.

User profile fields:
    user_id, age_group, occupation, gender,
    genre_preference_vector (19-D, recency-weighted),
    avg_rating, rating_volatility, n_ratings

Item profile fields:
    item_id, title, year, genres, popularity_tier, novelty_score

Outputs:
    user_profiles.json
    item_profiles.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "unknown",
]


# ────────────────────────────────────────────────────────────────────────────
#  Item profiles
# ────────────────────────────────────────────────────────────────────────────
def load_item_profiles(path: Path) -> dict:
    """Build per-item JSON profiles from u.item."""
    cols = ["item_id", "title", "release_date", "video_release", "url"] + GENRES
    items = pd.read_csv(path / "u.item", sep="|", names=cols, encoding="latin-1")
    items["item_id"] -= 1
    items["year"] = items["title"].str.extract(r"\((\d{4})\)").astype(float)

    ratings = pd.read_csv(path / "u.data", sep="\t",
                          names=["user_id", "item_id", "rating", "timestamp"])
    ratings["item_id"] -= 1
    item_counts = ratings["item_id"].value_counts()

    # Popularity tier by quartile
    q1, q2, q3 = item_counts.quantile([0.25, 0.5, 0.75])
    def tier(c):
        if c >= q3: return "blockbuster"
        if c >= q2: return "mainstream"
        if c >= q1: return "midlist"
        return "longtail"

    # Novelty: inverse popularity, normalised to [0, 1]
    pop = item_counts.reindex(items["item_id"], fill_value=0).values
    novelty = 1.0 - (pop / max(pop.max(), 1))

    profiles = {}
    for idx, row in items.iterrows():
        iid = int(row["item_id"])
        active_genres = [g for g in GENRES if row[g] == 1]
        profiles[iid] = {
            "item_id": iid,
            "title": str(row["title"]),
            "year": int(row["year"]) if not pd.isna(row["year"]) else None,
            "genres": active_genres,
            "n_ratings": int(item_counts.get(iid, 0)),
            "popularity_tier": tier(item_counts.get(iid, 0)),
            "novelty_score": float(novelty[idx]),
        }
    return profiles


# ────────────────────────────────────────────────────────────────────────────
#  User profiles
# ────────────────────────────────────────────────────────────────────────────
def recency_weights(timestamps: np.ndarray, half_life_days: float = 180) -> np.ndarray:
    """Exponential decay where each observation is weighted by its age."""
    ts = timestamps.astype(np.float64)
    age = ts.max() - ts
    age_days = age / 86400.0
    return np.exp(-np.log(2) * age_days / half_life_days)


def load_user_profiles(path: Path, item_profiles: dict) -> dict:
    """Build per-user JSON profiles from u.user and u.data."""
    users = pd.read_csv(
        path / "u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
    )
    users["user_id"] -= 1

    def age_group(a):
        if a < 18: return "under_18"
        if a < 25: return "18_24"
        if a < 35: return "25_34"
        if a < 50: return "35_49"
        return "50_plus"

    ratings = pd.read_csv(path / "u.data", sep="\t",
                          names=["user_id", "item_id", "rating", "timestamp"])
    ratings["user_id"] -= 1
    ratings["item_id"] -= 1

    profiles = {}
    for _, row in users.iterrows():
        uid = int(row["user_id"])
        u_rat = ratings[ratings["user_id"] == uid]
        if len(u_rat) == 0:
            genre_vec = np.zeros(len(GENRES))
        else:
            w = recency_weights(u_rat["timestamp"].values)
            genre_vec = np.zeros(len(GENRES))
            for r, item_id, weight in zip(u_rat["rating"].values,
                                          u_rat["item_id"].values, w):
                item_genres = item_profiles.get(int(item_id), {}).get("genres", [])
                for g in item_genres:
                    if g in GENRES:
                        genre_vec[GENRES.index(g)] += weight * (r - 3) / 2
            if np.abs(genre_vec).sum() > 0:
                genre_vec = genre_vec / np.abs(genre_vec).sum()

        profiles[uid] = {
            "user_id": uid,
            "age": int(row["age"]),
            "age_group": age_group(row["age"]),
            "gender": str(row["gender"]),
            "occupation": str(row["occupation"]),
            "n_ratings": int(len(u_rat)),
            "avg_rating": float(u_rat["rating"].mean()) if len(u_rat) else 0.0,
            "rating_volatility": float(u_rat["rating"].std()) if len(u_rat) > 1 else 0.0,
            "genre_preference_vector": [float(x) for x in genre_vec],
        }
    return profiles


# ────────────────────────────────────────────────────────────────────────────
#  Text rendering for downstream MiniLM
# ────────────────────────────────────────────────────────────────────────────
def render_user_text(prof: dict) -> str:
    top_genres = sorted(
        zip(GENRES, prof["genre_preference_vector"]),
        key=lambda x: -x[1],
    )[:3]
    top_str = ", ".join(g for g, _ in top_genres if _ > 0) or "none"
    return (
        f"A {prof['age_group'].replace('_', '-')} {prof['gender']} {prof['occupation']} "
        f"who has rated {prof['n_ratings']} films at average {prof['avg_rating']:.1f}/5. "
        f"Preferred genres: {top_str}."
    )


def render_item_text(prof: dict) -> str:
    g = ", ".join(prof["genres"]) or "uncategorised"
    yr = prof["year"] if prof["year"] else "unknown year"
    return (
        f"{prof['title']} ({yr}), a {prof['popularity_tier']} {g.lower()} film "
        f"with {prof['n_ratings']} ratings."
    )


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-100k", type=Path)
    parser.add_argument("--out", default="runs/stage2", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building item profiles...")
    items = load_item_profiles(args.data)
    with open(args.out / "item_profiles.json", "w") as f:
        json.dump(items, f, indent=2)

    print("Building user profiles...")
    users = load_user_profiles(args.data, items)
    with open(args.out / "user_profiles.json", "w") as f:
        json.dump(users, f, indent=2)

    print(f"Stage 2 complete. {len(users)} users, {len(items)} items written to {args.out}")


if __name__ == "__main__":
    main()
