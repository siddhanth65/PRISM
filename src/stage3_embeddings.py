"""
Stage 3: MiniLM embeddings and positive-prototype computation.

Three embedding tables are produced:
    1. User embeddings:  one 384-D vector per user (from rendered user text)
    2. Item embeddings:  one 384-D vector per item (from rendered item text)
    3. Positive prototype: the L2-normalised centroid of pair embeddings
       from training pairs that crossed the user's personalised threshold.

The pair-embedding idea is intentionally novel: instead of just concatenating
a user vector with an item vector, we encode the interaction itself as a
sentence ("This user, watching this movie, gives a rating of X"), pass it
through MiniLM, and compute similarity against the positive prototype. This
turns Stage 4's similarity score into a learned semantic match feature.

Outputs:
    user_embeddings.npy        (n_users, 384)
    item_embeddings.npy        (n_items, 384)
    positive_prototype.npy     (384,)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from stage2_profiles import render_item_text, render_user_text


# ────────────────────────────────────────────────────────────────────────────
#  Encoding helpers
# ────────────────────────────────────────────────────────────────────────────
def encode_batch(model: SentenceTransformer, texts, batch_size=64) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


def render_pair_text(u: dict, i: dict, rating: float) -> str:
    """Render a (user, item, rating) tuple as a single sentence for MiniLM."""
    u_text = render_user_text(u)
    i_text = render_item_text(i)
    return f"{u_text} The user watches {i_text} and rates it {rating:.1f} out of 5."


# ────────────────────────────────────────────────────────────────────────────
#  Positive-prototype centroid
# ────────────────────────────────────────────────────────────────────────────
def build_positive_prototype(model, df, user_profs, item_profs,
                             max_samples=5000, batch_size=64) -> np.ndarray:
    """Centroid of pair embeddings from positive training pairs.

    Stage 4 uses cosine similarity to this prototype as a feature: candidates
    whose pair embedding lies close to the centroid of liked pairs are more
    likely to be liked themselves.
    """
    user_means = df.groupby("user_id")["rating"].mean().to_dict()
    pos_mask = df["rating"].values >= df["user_id"].map(user_means).values
    pos = df[pos_mask]
    if len(pos) > max_samples:
        pos = pos.sample(max_samples, random_state=42)

    texts = []
    for _, row in tqdm(pos.iterrows(), total=len(pos), desc="Pair sentences"):
        u_prof = user_profs[int(row["user_id"])]
        i_prof = item_profs[int(row["item_id"])]
        texts.append(render_pair_text(u_prof, i_prof, row["rating"]))

    pair_embs = encode_batch(model, texts, batch_size=batch_size)
    centroid = pair_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    return centroid.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-100k", type=Path)
    parser.add_argument("--profiles", default="runs/stage2", type=Path)
    parser.add_argument("--out", default="runs/stage3", type=Path)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.profiles / "user_profiles.json") as f:
        user_profs = {int(k): v for k, v in json.load(f).items()}
    with open(args.profiles / "item_profiles.json") as f:
        item_profs = {int(k): v for k, v in json.load(f).items()}

    df = pd.read_csv(args.data / "u.data", sep="\t",
                     names=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"] -= 1
    df["item_id"] -= 1

    print(f"Loading {args.model}...")
    model = SentenceTransformer(args.model, device=args.device)

    n_users = max(user_profs) + 1
    n_items = max(item_profs) + 1

    print("Encoding user texts...")
    user_texts = [render_user_text(user_profs[u]) for u in range(n_users)]
    user_embs = encode_batch(model, user_texts)
    np.save(args.out / "user_embeddings.npy", user_embs)

    print("Encoding item texts...")
    item_texts = [render_item_text(item_profs[i]) for i in range(n_items)]
    item_embs = encode_batch(model, item_texts)
    np.save(args.out / "item_embeddings.npy", item_embs)

    print("Computing positive prototype...")
    prototype = build_positive_prototype(model, df, user_profs, item_profs)
    np.save(args.out / "positive_prototype.npy", prototype)

    print(f"Stage 3 complete. Embeddings at {args.out}")


if __name__ == "__main__":
    main()
