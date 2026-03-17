# Content Recommendation System with Embeddings & Ranking (2025)

End-to-end recommendation pipeline in Python:
- Neural retrieval model (TensorFlow) to learn user/item embeddings
- FAISS nearest-neighbor retrieval for candidate generation
- Gradient-boosted re-ranker for final ranking
- Evaluation with `NDCG@10` and `Recall@K`

## Files
- `content_recommendation_system_2025.py`: complete runnable pipeline
- `requirements.txt`: dependencies

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (auto-download dataset)
By default, the script downloads **MovieLens 100K** automatically into `data/`.

```bash
python3 content_recommendation_system_2025.py
```

## Useful options
```bash
# Re-download dataset from scratch
python3 content_recommendation_system_2025.py --force-download

# Change implicit-feedback threshold
python3 content_recommendation_system_2025.py --min-rating 3.5

# Use synthetic data instead of MovieLens
python3 content_recommendation_system_2025.py --dataset-source synthetic
```

## Example output
```text
=== Content Recommendation System with Embeddings & Ranking ===
Dataset source: movielens
Users: ... , Items: ...
Train interactions: ... , Test users: ...
FAISS enabled: True
Recall@50: 0.XXXX
NDCG@10: 0.XXXX
```

## Notes
- If `faiss-cpu` is unavailable, retrieval falls back to brute-force cosine similarity.
- The script treats ratings `>= min_rating` as positive implicit interactions.
