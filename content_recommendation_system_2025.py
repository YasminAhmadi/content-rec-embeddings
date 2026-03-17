"""
Content Recommendation System with Embeddings & Ranking (2025 style)

Pipeline:
1) Train a neural retrieval model (TensorFlow) on user-item interactions.
2) Generate user/item embeddings.
3) Retrieve nearest-neighbor candidates with FAISS.
4) Train a gradient-boosted re-ranker.
5) Evaluate with NDCG@10 and Recall@K.

By default, the script downloads MovieLens 100K automatically so no local dataset
is required.

Suggested install:
    pip install tensorflow faiss-cpu scikit-learn numpy
"""

from __future__ import annotations

import argparse
import math
import random
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False



# Utilities


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)



# Synthetic data generation


def load_movielens_100k(
    data_dir: Path,
    min_rating: float,
    force_download: bool,
) -> Tuple[List[Tuple[int, int]], int, int]:
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = data_dir / "ml-100k.zip"
    extracted_root = data_dir / "ml-100k"
    ratings_file = extracted_root / "u.data"

    if force_download and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(f"Downloading MovieLens 100K from {url} ...")
        urllib.request.urlretrieve(url, zip_path)

    if force_download and extracted_root.exists():
        for p in sorted(extracted_root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        extracted_root.rmdir()

    if not ratings_file.exists():
        print(f"Extracting dataset to {data_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    if not ratings_file.exists():
        raise FileNotFoundError(f"Could not locate ratings file: {ratings_file}")

    # Keep only positive interactions for implicit-feedback training.
    raw_interactions: List[Tuple[str, str]] = []
    with ratings_file.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            user_id, item_id, rating_str = parts[0], parts[1], parts[2]
            if float(rating_str) >= min_rating:
                raw_interactions.append((user_id, item_id))

    user_to_items: Dict[str, Set[str]] = defaultdict(set)
    for user_id, item_id in raw_interactions:
        user_to_items[user_id].add(item_id)

    # Ensure each user has enough history for leave-one-out splitting.
    kept_users = {u for u, items in user_to_items.items() if len(items) >= 2}
    filtered = [(u, i) for (u, i) in raw_interactions if u in kept_users]

    if not filtered:
        raise ValueError("No interactions left after filtering; try lowering --min-rating")

    unique_users = sorted({u for u, _ in filtered})
    unique_items = sorted({i for _, i in filtered})
    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {i: idx for idx, i in enumerate(unique_items)}

    mapped = [(user_map[u], item_map[i]) for u, i in filtered]
    return mapped, len(unique_users), len(unique_items)

def generate_synthetic_interactions(
    num_users: int,
    num_items: int,
    true_dim: int,
    positives_per_user: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)

    # Hidden ground-truth factors define user preferences over items.
    user_latent = rng.normal(0, 1, size=(num_users, true_dim))
    item_latent = rng.normal(0, 1, size=(num_items, true_dim))

    interactions: List[Tuple[int, int]] = []
    for u in range(num_users):
        scores = user_latent[u] @ item_latent.T
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()

        chosen = rng.choice(
            np.arange(num_items),
            size=min(positives_per_user, num_items),
            replace=False,
            p=probs,
        )
        for i in chosen:
            interactions.append((u, int(i)))

    return interactions


def leave_one_out_split(
    interactions: Sequence[Tuple[int, int]],
    num_users: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, Set[int]]]:
    rng = np.random.default_rng(seed)

    by_user: Dict[int, List[int]] = {u: [] for u in range(num_users)}
    for u, i in interactions:
        by_user[u].append(i)

    train: List[Tuple[int, int]] = []
    test_item_by_user: Dict[int, int] = {}
    train_seen_by_user: Dict[int, Set[int]] = {u: set() for u in range(num_users)}

    for u, items in by_user.items():
        if not items:
            continue
        holdout_idx = int(rng.integers(0, len(items)))
        holdout_item = items[holdout_idx]
        test_item_by_user[u] = holdout_item

        for idx, it in enumerate(items):
            if idx == holdout_idx:
                continue
            train.append((u, it))
            train_seen_by_user[u].add(it)

    return train, test_item_by_user, train_seen_by_user


def build_pairwise_training_data(
    train_interactions: Sequence[Tuple[int, int]],
    num_items: int,
    negatives_per_positive: int,
    train_seen_by_user: Dict[int, Set[int]],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    users: List[int] = []
    items: List[int] = []
    labels: List[int] = []

    for u, pos_i in train_interactions:
        users.append(u)
        items.append(pos_i)
        labels.append(1)

        seen = train_seen_by_user[u]
        for _ in range(negatives_per_positive):
            neg_i = int(rng.integers(0, num_items))
            while neg_i in seen or neg_i == pos_i:
                neg_i = int(rng.integers(0, num_items))
            users.append(u)
            items.append(neg_i)
            labels.append(0)

    return np.array(users), np.array(items), np.array(labels)



# Neural retrieval model


class RetrievalModel(tf.keras.Model):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, inputs, training: bool = False):  # type: ignore
        user_ids, item_ids = inputs
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        logits = tf.reduce_sum(u * i, axis=1)
        return logits


def train_retriever(
    num_users: int,
    num_items: int,
    embedding_dim: int,
    users: np.ndarray,
    items: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    epochs: int,
) -> RetrievalModel:
    model = RetrievalModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )

    model.fit(
        x=(users, items),
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        shuffle=True,
    )
    return model



# Candidate retrieval


@dataclass
class RetrievalOutput:
    candidates_by_user: Dict[int, List[int]]
    scores_by_user: Dict[int, List[float]]


def retrieve_candidates(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    train_seen_by_user: Dict[int, Set[int]],
    top_k: int,
) -> RetrievalOutput:
    user_vecs = l2_normalize(user_vectors.astype(np.float32))
    item_vecs = l2_normalize(item_vectors.astype(np.float32))

    candidates_by_user: Dict[int, List[int]] = {}
    scores_by_user: Dict[int, List[float]] = {}

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(item_vecs.shape[1])
        index.add(item_vecs)

        # Search a little deeper so we can drop seen items and still keep top_k.
        search_k = min(item_vecs.shape[0], top_k + 100)
        sim, idx = index.search(user_vecs, search_k)

        for u in range(user_vecs.shape[0]):
            seen = train_seen_by_user.get(u, set())
            cand: List[int] = []
            cand_scores: List[float] = []
            for rank in range(search_k):
                i = int(idx[u, rank])
                if i in seen:
                    continue
                cand.append(i)
                cand_scores.append(float(sim[u, rank]))
                if len(cand) >= top_k:
                    break
            candidates_by_user[u] = cand
            scores_by_user[u] = cand_scores
    else:
        # Fallback: brute-force cosine similarity if FAISS is unavailable.
        sim = user_vecs @ item_vecs.T
        for u in range(user_vecs.shape[0]):
            seen = train_seen_by_user.get(u, set())
            ranking = np.argsort(-sim[u])
            cand: List[int] = []
            cand_scores: List[float] = []
            for i in ranking:
                ii = int(i)
                if ii in seen:
                    continue
                cand.append(ii)
                cand_scores.append(float(sim[u, ii]))
                if len(cand) >= top_k:
                    break
            candidates_by_user[u] = cand
            scores_by_user[u] = cand_scores

    return RetrievalOutput(candidates_by_user=candidates_by_user, scores_by_user=scores_by_user)



# Re-ranker


def build_item_popularity(train_interactions: Sequence[Tuple[int, int]], num_items: int) -> np.ndarray:
    pop = np.zeros(num_items, dtype=np.float32)
    for _, i in train_interactions:
        pop[i] += 1.0
    if pop.max() > 0:
        pop = pop / pop.max()
    return pop


def make_features(
    user_vec: np.ndarray,
    item_vec: np.ndarray,
    base_score: float,
    item_popularity: float,
    user_activity: float,
) -> List[float]:
    dot = float(np.dot(user_vec, item_vec))
    diff_l2 = float(np.linalg.norm(user_vec - item_vec))
    return [dot, base_score, diff_l2, item_popularity, user_activity]


def build_reranker_data(
    retrieval: RetrievalOutput,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    item_popularity: np.ndarray,
    train_seen_by_user: Dict[int, Set[int]],
    test_item_by_user: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []

    max_activity = max((len(s) for s in train_seen_by_user.values()), default=1)

    for u, true_item in test_item_by_user.items():
        user_activity = len(train_seen_by_user.get(u, set())) / max(1, max_activity)
        cands = retrieval.candidates_by_user.get(u, [])
        scores = retrieval.scores_by_user.get(u, [])

        # Ensure the positive example is present for supervised re-ranking.
        if true_item not in cands:
            cands = cands + [true_item]
            true_score = float(np.dot(user_vectors[u], item_vectors[true_item]))
            scores = scores + [true_score]

        for item_id, base_score in zip(cands, scores):
            feat = make_features(
                user_vec=user_vectors[u],
                item_vec=item_vectors[item_id],
                base_score=base_score,
                item_popularity=float(item_popularity[item_id]),
                user_activity=float(user_activity),
            )
            X.append(feat)
            y.append(1 if item_id == true_item else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)



# Metrics


def recall_at_k(rankings: Dict[int, List[int]], truth: Dict[int, int], k: int) -> float:
    hits = 0
    total = 0
    for u, true_item in truth.items():
        recs = rankings.get(u, [])[:k]
        hits += int(true_item in recs)
        total += 1
    return hits / max(total, 1)


def ndcg_at_k(rankings: Dict[int, List[int]], truth: Dict[int, int], k: int) -> float:
    # With one relevant item per user, DCG is 1/log2(rank+1) when hit; IDCG is 1.
    total = 0.0
    n = 0
    for u, true_item in truth.items():
        recs = rankings.get(u, [])[:k]
        gain = 0.0
        for rank, item_id in enumerate(recs, start=1):
            if item_id == true_item:
                gain = 1.0 / math.log2(rank + 1)
                break
        total += gain
        n += 1
    return total / max(n, 1)



# Full pipeline


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.dataset_source == "movielens":
        interactions, num_users, num_items = load_movielens_100k(
            data_dir=Path(args.data_dir),
            min_rating=args.min_rating,
            force_download=args.force_download,
        )
    else:
        interactions = generate_synthetic_interactions(
            num_users=args.num_users,
            num_items=args.num_items,
            true_dim=args.true_latent_dim,
            positives_per_user=args.positives_per_user,
            seed=args.seed,
        )
        num_users = args.num_users
        num_items = args.num_items

    train, test_item_by_user, train_seen_by_user = leave_one_out_split(
        interactions=interactions,
        num_users=num_users,
        seed=args.seed + 1,
    )

    users, items, labels = build_pairwise_training_data(
        train_interactions=train,
        num_items=num_items,
        negatives_per_positive=args.negatives_per_positive,
        train_seen_by_user=train_seen_by_user,
        seed=args.seed + 2,
    )

    retriever = train_retriever(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        users=users,
        items=items,
        labels=labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    user_vectors = retriever.user_embedding.get_weights()[0]
    item_vectors = retriever.item_embedding.get_weights()[0]

    retrieval = retrieve_candidates(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        train_seen_by_user=train_seen_by_user,
        top_k=args.candidate_k,
    )

    item_popularity = build_item_popularity(train_interactions=train, num_items=num_items)

    X_train, y_train = build_reranker_data(
        retrieval=retrieval,
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        item_popularity=item_popularity,
        train_seen_by_user=train_seen_by_user,
        test_item_by_user=test_item_by_user,
    )

    reranker = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=args.seed,
    )
    reranker.fit(X_train, y_train)

    # Build final rankings after re-ranking.
    final_rankings: Dict[int, List[int]] = {}
    max_activity = max((len(s) for s in train_seen_by_user.values()), default=1)

    for u, cands in retrieval.candidates_by_user.items():
        if not cands:
            final_rankings[u] = []
            continue

        user_activity = len(train_seen_by_user.get(u, set())) / max(1, max_activity)
        scores = retrieval.scores_by_user[u]

        X_user = np.array(
            [
                make_features(
                    user_vec=user_vectors[u],
                    item_vec=item_vectors[item_id],
                    base_score=base_score,
                    item_popularity=float(item_popularity[item_id]),
                    user_activity=float(user_activity),
                )
                for item_id, base_score in zip(cands, scores)
            ],
            dtype=np.float32,
        )

        proba = reranker.predict_proba(X_user)[:, 1]
        reranked = [item for item, _ in sorted(zip(cands, proba), key=lambda t: t[1], reverse=True)]
        final_rankings[u] = reranked

    recall_k = recall_at_k(final_rankings, test_item_by_user, k=args.recall_k)
    ndcg_10 = ndcg_at_k(final_rankings, test_item_by_user, k=10)

    print("=== Content Recommendation System with Embeddings & Ranking ===")
    print(f"Dataset source: {args.dataset_source}")
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Train interactions: {len(train)}, Test users: {len(test_item_by_user)}")
    print(f"FAISS enabled: {FAISS_AVAILABLE}")
    print(f"Recall@{args.recall_k}: {recall_k:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural retrieval + FAISS + GBDT re-ranking demo")
    parser.add_argument("--dataset-source", type=str, default="movielens", choices=["movielens", "synthetic"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--num-users", type=int, default=600)
    parser.add_argument("--num-items", type=int, default=2000)
    parser.add_argument("--true-latent-dim", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--positives-per-user", type=int, default=40)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
