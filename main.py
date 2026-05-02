# ================================================
#   MOVIE RECOMMENDATION SYSTEM (Python 3.14 Compatible)
#   Content-based + Simple SVD Collaborative + Hybrid
# ================================================

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

print("Loading data...")
movies = pd.read_csv('Data/movies.csv')
ratings = pd.read_csv('Data/ratings.csv')

MAX_USER_ID = ratings['userId'].max()

print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")


# ====================== CONTENT-BASED ======================
print("Building Content-Based Model...")

movies['genres_clean'] = movies['genres'].str.replace('|', ' ')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

os.makedirs('models', exist_ok=True)
np.save('models/cosine_sim.npy', cosine_sim)
movies.to_pickle('models/movies.pkl')

def content_based_recommend(title, top_n=10):
    if title not in movies['title'].values:
        return pd.DataFrame({"Error": [f"Movie '{title}' not found!"]})
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]


# ====================== COLLABORATIVE FILTERING (Simple SVD) ======================
print("Training Simple SVD Collaborative Model...")

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item_matrix.values

# Normalize (subtract mean) - only consider rated items (non-zero)
ratings_count = (matrix != 0).sum(axis=1)
ratings_sum = matrix.sum(axis=1)
user_means = np.divide(ratings_sum, ratings_count, out=np.zeros_like(ratings_sum, dtype=float), where=ratings_count!=0)
matrix_normalized = np.where(matrix != 0, matrix - user_means.reshape(-1, 1), 0)


# Apply SVD
k = 50  # number of latent factors
U, sigma, Vt = svds(matrix_normalized, k=k)
sigma = np.diag(sigma)

# Predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
predicted_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def collaborative_recommend(user_id, top_n=10):
    if user_id not in predicted_df.index:
        return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
    
    user_predictions = predicted_df.loc[user_id]
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    # Remove already rated movies
    candidates = user_predictions.drop(user_rated, errors='ignore')
    top_movie_ids = candidates.nlargest(top_n).index
    
    return movies[movies['movieId'].isin(top_movie_ids)][['title', 'genres']]


# ====================== HYBRID RECOMMENDATION ======================
def hybrid_recommend(user_id, top_n=10, alpha=0.65):
    if user_id not in predicted_df.index:
        return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
    
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    candidates = movies[~movies['movieId'].isin(user_rated)].copy()
    
    # Collaborative scores
    collab_scores = []
    for mid in candidates['movieId']:
        if mid in predicted_df.columns:
            score = predicted_df.loc[user_id, mid]
        else:
            score = 3.0
        collab_scores.append(score)
    
    # Content-based scores from highly rated movies
    high_rated = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4.0)]['movieId']
    if len(high_rated) == 0:
        high_rated = ratings[ratings['userId'] == user_id]['movieId'].head(5)
    
    content_scores = np.zeros(len(candidates))
    for mid in high_rated:
        if mid in movies['movieId'].values:
            idx = movies[movies['movieId'] == mid].index[0]
            content_scores += cosine_sim[idx][candidates.index]
    
    content_scores = content_scores / max(len(high_rated), 1)
    
    # Hybrid score
    final_scores = alpha * np.array(collab_scores) + (1 - alpha) * content_scores
    candidates['hybrid_score'] = final_scores
    
    recommendations = candidates.sort_values('hybrid_score', ascending=False).head(top_n)
    return recommendations[['title', 'genres', 'hybrid_score']]



# ====================== TEST ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM READY (Python 3.14 Compatible)")
    print("="*60)
    
    print("\n1. Content-Based Recommendation:")
    print(content_based_recommend("Toy Story (1995)", 5))
    
    print("\n2. Collaborative Recommendation for User 1:")
    print(collaborative_recommend(user_id=1, top_n=5))
    
    print("\n3. Hybrid Recommendation for User 1:")
    print(hybrid_recommend(user_id=1, top_n=8, alpha=0.65))
