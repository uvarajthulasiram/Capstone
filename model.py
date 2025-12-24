import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load saved artifacts
# -----------------------------
with open("artifacts/sentiment_pipeline.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("artifacts/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

z = np.load("artifacts/user_topk_sim.npz", allow_pickle=True)
users = z["users"]
items = z["items"]
topk_indices = z["topk_indices"] 
topk_sims = z["topk_sims"] 

# Map user -> row index
user_to_idx = {u: i for i, u in enumerate(users)}

# -----------------------------
# Sentiment Prediction
# -----------------------------
def predict_sentiment(text):
    """
    Predict sentiment for a single review
    """
    return int(sentiment_model.predict([text])[0])

# -----------------------------
# User-User Recommendation (Top-K)
# -----------------------------
def recommend_user(user, top_n=5):
    """
    Recommend products for a given user using Top-K user-user collaborative filtering
    """
    if user not in user_to_idx:
        return []

    u_idx = user_to_idx[user]

    nbr_idx = topk_indices[u_idx]
    nbr_sims = topk_sims[u_idx]

    denom = float(nbr_sims.sum())
    if denom == 0.0:
        return []

    # Weighted prediction over items
    neighbor_ratings = user_item_matrix.values[nbr_idx, :] 
    scores = (nbr_sims @ neighbor_ratings) / denom 

    # Remove already-rated items for this user
    user_rated = user_item_matrix.values[u_idx, :]
    scores = np.where(user_rated == 0, scores, -np.inf)

    # Top-N item indices
    if top_n <= 0:
        return []

    top_idx = np.argpartition(-scores, range(min(top_n, scores.size)))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return items[top_idx].tolist()
