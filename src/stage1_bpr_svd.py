"""
Stage 1: BPR-MF and multi-scale SVD.

Trains a Bayesian Personalised Ranking matrix factorisation model and computes
SVD scores at three latent dimensions (k=20, 50, 100). BPR uses popularity-biased
negative sampling to produce harder, more informative gradient updates than the
uniform-sampling default. SVD scores are not used directly as the ranking signal;
they are passed downstream as features for the Stage-6 reranker.

Outputs:
    bpr_scores.npy        (n_users, n_items) BPR-MF score matrix
    svd_k20.npy           (n_users, n_items) SVD reconstruction at k=20
    svd_k50.npy           same at k=50
    svd_k100.npy          same at k=100
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
#  Data loading
# ────────────────────────────────────────────────────────────────────────────
def load_ml100k(path: Path):
    """Load MovieLens 100K ratings as a DataFrame with 0-indexed user/item IDs."""
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path / "u.data", sep="\t", names=cols)
    df["user_id"] -= 1
    df["item_id"] -= 1
    return df


def build_user_means(df: pd.DataFrame) -> dict:
    """Per-user average rating, used downstream as the personalised threshold."""
    return df.groupby("user_id")["rating"].mean().to_dict()


def positive_pairs(df: pd.DataFrame, user_means: dict) -> np.ndarray:
    """A pair is positive if rating >= the user's own mean rating."""
    mask = df["rating"].values >= df["user_id"].map(user_means).values
    return df[mask][["user_id", "item_id"]].to_numpy()


# ────────────────────────────────────────────────────────────────────────────
#  BPR-MF model
# ────────────────────────────────────────────────────────────────────────────
class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)

    def forward(self, users, pos_items, neg_items):
        pos_scores = self.score(users, pos_items)
        neg_scores = self.score(users, neg_items)
        return pos_scores - neg_scores


class BPRDataset(Dataset):
    """Yields (user, positive_item, negative_item) triplets.

    Negative items are sampled with probability proportional to popularity ** 0.75,
    forcing the model to discriminate against items the user might plausibly know,
    rather than easy long-tail items.
    """

    def __init__(self, positives: np.ndarray, n_items: int, item_pop: np.ndarray):
        self.positives = positives
        self.n_items = n_items
        self.user_pos = {}
        for u, i in positives:
            self.user_pos.setdefault(int(u), set()).add(int(i))
        pop = np.power(item_pop.astype(np.float64) + 1, 0.75)
        self.neg_p = pop / pop.sum()

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        u, pos = self.positives[idx]
        seen = self.user_pos[int(u)]
        while True:
            neg = np.random.choice(self.n_items, p=self.neg_p)
            if neg not in seen:
                return int(u), int(pos), int(neg)


def train_bpr(df, n_users, n_items, n_factors=128, epochs=50, batch_size=4096,
              lr=1e-3, device="cpu"):
    user_means = build_user_means(df)
    pos = positive_pairs(df, user_means)
    item_pop = df["item_id"].value_counts().reindex(range(n_items), fill_value=0).values

    ds = BPRDataset(pos, n_items, item_pop)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BPRMF(n_users, n_items, n_factors).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for u, p, n in tqdm(dl, desc=f"BPR epoch {ep+1}/{epochs}", leave=False):
            u, p, n = u.to(device), p.to(device), n.to(device)
            diff = model(u, p, n)
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * u.size(0)
        sched.step()
        print(f"  epoch {ep+1}: loss={ep_loss / len(ds):.4f}")

    model.eval()
    with torch.no_grad():
        U = model.user_emb.weight.detach().cpu().numpy()
        V = model.item_emb.weight.detach().cpu().numpy()
    return U @ V.T


# ────────────────────────────────────────────────────────────────────────────
#  Multi-scale SVD
# ────────────────────────────────────────────────────────────────────────────
def multi_scale_svd(df, n_users, n_items, ks=(20, 50, 100)):
    R = csr_matrix(
        (df["rating"].astype(float), (df["user_id"], df["item_id"])),
        shape=(n_users, n_items),
    )
    results = {}
    for k in ks:
        U, s, Vt = svds(R.astype(float), k=k)
        results[k] = (U * s) @ Vt
        print(f"  SVD k={k}: ||X̂||_F = {np.linalg.norm(results[k]):.2f}")
    return results


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-100k", type=Path)
    parser.add_argument("--out", default="runs/stage1", type=Path)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_ml100k(args.data)
    n_users = df["user_id"].max() + 1
    n_items = df["item_id"].max() + 1
    print(f"Loaded {len(df)} ratings, {n_users} users, {n_items} items")

    print("Training BPR-MF...")
    bpr_scores = train_bpr(df, n_users, n_items,
                           n_factors=args.factors, epochs=args.epochs,
                           device=args.device)
    np.save(args.out / "bpr_scores.npy", bpr_scores)

    print("Computing multi-scale SVD...")
    svd_results = multi_scale_svd(df, n_users, n_items)
    for k, mat in svd_results.items():
        np.save(args.out / f"svd_k{k}.npy", mat)

    print(f"Stage 1 complete. Artefacts in {args.out}")


if __name__ == "__main__":
    main()
