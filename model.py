import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Load saved artifacts
# -----------------------------
with open("artifacts/sentiment_pipeline.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("artifacts/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("artifacts/user_sim_df.pkl", "rb") as f:
    user_sim_df = pickle.load(f)


# -----------------------------
# Sentiment Prediction
# -----------------------------
def predict_sentiment(text):
    """
    Predict sentiment for a single review
    """
    return int(sentiment_model.predict([text])[0])


# -----------------------------
# User-User Recommendation
# -----------------------------
def recommend_user(user, top_n=5):
    """
    Recommend products for a given user using user-user collaborative filtering
    """
    if user not in user_item_matrix.index:
        return []

    # Similarity scores for the user
    sim_scores = user_sim_df.loc[user]

    # Weighted sum of ratings
    weighted_ratings = user_item_matrix.T.dot(sim_scores) / sim_scores.sum()

    # Remove already rated items
    rated_items = user_item_matrix.loc[user]
    recommendations = weighted_ratings[rated_items == 0]

    return recommendations.sort_values(ascending=False).head(top_n).index.tolist()
