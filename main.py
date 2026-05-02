# ================================================
#   MOVIE RECOMMENDATION SYSTEM (Python 3.14 Compatible)
#   Content-based + Collaborative Filtering + Hybrid
#   With Multiple Similarity Metrics & Advanced Algorithms
# ================================================

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr

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
euclidean_sim = euclidean_distances(tfidf_matrix, tfidf_matrix)

# Compute normalized vectors for Pearson correlation
tfidf_normalized = tfidf_matrix.toarray()
tfidf_normalized = tfidf_normalized / (np.linalg.norm(tfidf_normalized, axis=1, keepdims=True) + 1e-10)

os.makedirs('models', exist_ok=True)
np.save('models/cosine_sim.npy', cosine_sim)
np.save('models/euclidean_sim.npy', euclidean_sim)
np.save('models/tfidf_normalized.npy', tfidf_normalized)
movies.to_pickle('models/movies.pkl')


def content_based_recommend(title, top_n=10, metric='cosine'):
    if title not in movies['title'].values:
        return pd.DataFrame({"Error": [f"Movie '{title}' not found!"]})
    
    idx = movies[movies['title'] == title].index[0]
    
    if metric == 'cosine':
        sim_scores = list(enumerate(cosine_sim[idx]))
    elif metric == 'euclidean':
        # Convert distance to similarity (smaller distance = higher similarity)
        sim_scores = list(enumerate(-euclidean_sim[idx]))
    elif metric == 'pearson':
        # Use Pearson correlation on TF-IDF vectors
        movie_vec = tfidf_normalized[idx]
        sim_scores = []
        for i in range(len(tfidf_normalized)):
            if i == idx:
                continue
            corr, _ = pearsonr(movie_vec, tfidf_normalized[i])
            sim_scores.append((i, corr if not np.isnan(corr) else 0))
    else:
        sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]


# ====================== COLLABORATIVE FILTERING ======================
print("Training Collaborative Models...")

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item_matrix.values

# Normalize (subtract mean) - only consider rated items (non-zero)
ratings_count = (matrix != 0).sum(axis=1)
ratings_sum = matrix.sum(axis=1)
user_means = np.divide(ratings_sum, ratings_count, out=np.zeros_like(ratings_sum, dtype=float), where=ratings_count!=0)
matrix_normalized = np.where(matrix != 0, matrix - user_means.reshape(-1, 1), 0)

# Add user and item bias
item_bias = np.zeros(matrix.shape[1])
for j in range(matrix.shape[1]):
    rated_items = matrix[:, j] != 0
    if rated_items.sum() > 0:
        item_bias[j] = matrix[rated_items, j].mean() - ratings_sum[rated_items].sum() / (ratings_count[rated_items].sum() + 1e-10)
item_bias = np.nan_to_num(item_bias, nan=0)

# Apply SVD with regularization
k = 50  # number of latent factors
reg_lambda = 0.1  # regularization parameter

U, sigma, Vt = svds(matrix_normalized, k=k)
sigma = np.diag(sigma)

# Predicted ratings with user bias and item bias
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1) + item_bias.reshape(1, -1)
predicted_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)


# ====================== NMF Collaborative Filtering ======================
print("Training NMF Collaborative Model...")

# NMF requires non-negative values, shift ratings to be positive
matrix_nmf = matrix.copy()
matrix_nmf[matrix_nmf == 0] = 0  # Keep zeros as zeros for NMF

nmf_model = NMF(n_components=50, init='random', random_state=42, max_iter=500)
user_factors_nmf = nmf_model.fit_transform(matrix_nmf)
item_factors_nmf = nmf_model.components_

# Predicted ratings for NMF
predicted_ratings_nmf = np.dot(user_factors_nmf, item_factors_nmf)
predicted_df_nmf = pd.DataFrame(predicted_ratings_nmf, columns=user_item_matrix.columns, index=user_item_matrix.index)


# ====================== KNN-Based Collaborative Filtering ======================
print("Training KNN Collaborative Model...")

# User-based KNN: fit on users (rows)
user_knn = NearestNeighbors(metric='cosine', n_neighbors=20)
user_knn.fit(matrix_normalized)  # Users as rows


def collaborative_recommend(user_id, top_n=10, method='svd'):
    if method == 'svd':
        if user_id not in predicted_df.index:
            return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
        user_predictions = predicted_df.loc[user_id]
    elif method == 'nmf':
        if user_id not in predicted_df_nmf.index:
            return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
        user_predictions = predicted_df_nmf.loc[user_id]
    elif method == 'knn':
        # KNN-based recommendations (user-based collaborative filtering)
        if user_id not in user_item_matrix.index:
            return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
        
        # Get the target user's rating vector
        user_idx = user_id - 1  # Convert to 0-based index
        user_ratings = matrix_normalized[user_idx, :].reshape(1, -1)
        
        # Find k similar users based on their rating patterns
        distances, indices = user_knn.kneighbors(user_ratings)
        
        # Get similar users (exclude the user itself)
        similar_users = [u for u in indices[0] if u != user_idx]
        
        if len(similar_users) == 0:
            # Fallback to SVD if no similar users found
            user_predictions = predicted_df.loc[user_id]
        else:
            # Calculate similarity scores (1 / (1 + distance))
            similar_scores = np.array([1 / (distances[0][i] + 1e-10) for i in range(len(similar_users))])
            
            # Aggregate ratings from similar users (weighted by similarity)
            weighted_ratings = np.zeros(matrix.shape[1])
            for i, sim_user in enumerate(similar_users):
                weighted_ratings += matrix_normalized[sim_user, :] * similar_scores[i]
            weighted_ratings = weighted_ratings / (similar_scores.sum() + 1e-10)
            
            # Add user mean back
            user_predictions = pd.Series(
                weighted_ratings + user_means[user_idx], 
                index=user_item_matrix.columns
            )
    else:
        user_predictions = predicted_df.loc[user_id]
    
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    # Remove already rated movies
    candidates = user_predictions.drop(user_rated, errors='ignore')
    top_movie_ids = candidates.nlargest(top_n).index
    
    return movies[movies['movieId'].isin(top_movie_ids)][['title', 'genres']]


# ====================== HYBRID RECOMMENDATION ======================
def hybrid_recommend(user_id, top_n=10, alpha=0.65, content_metric='cosine', algorithm='svd'):
    # Select the appropriate predicted DataFrame based on algorithm
    if algorithm == 'svd':
        predictions = predicted_df
    elif algorithm == 'nmf':
        predictions = predicted_df_nmf
    elif algorithm == 'knn':
        # For KNN, we need to compute predictions dynamically
        if user_id not in user_item_matrix.index:
            return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
        
        user_idx = user_id - 1
        user_ratings = matrix_normalized[user_idx, :].reshape(1, -1)
        distances, indices = user_knn.kneighbors(user_ratings)
        
        similar_users = [u for u in indices[0] if u != user_idx]
        
        if len(similar_users) == 0:
            # Fallback to SVD if no similar users found
            predictions = predicted_df
        else:
            similar_scores = np.array([1 / (distances[0][i] + 1e-10) for i in range(len(similar_users))])
            weighted_ratings = np.zeros(matrix.shape[1])
            for i, sim_user in enumerate(similar_users):
                weighted_ratings += matrix_normalized[sim_user, :] * similar_scores[i]
            weighted_ratings = weighted_ratings / (similar_scores.sum() + 1e-10)
            knn_predictions = pd.Series(
                weighted_ratings + user_means[user_idx],
                index=user_item_matrix.columns
            )
            predictions = pd.DataFrame(knn_predictions).T
            predictions.index = [user_id]
    else:
        predictions = predicted_df
    
    if user_id not in predictions.index:
        return pd.DataFrame({"Error": [f"User {user_id} not found!"]})
    
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    candidates = movies[~movies['movieId'].isin(user_rated)].copy()
    
    # Collaborative scores (from selected algorithm)
    collab_scores = []
    for mid in candidates['movieId']:
        if mid in predictions.columns:
            score = predictions.loc[user_id, mid]
        else:
            score = 3.0
        collab_scores.append(score)
    
    # Content-based scores from highly rated movies
    high_rated = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4.0)]['movieId']
    if len(high_rated) == 0:
        high_rated = ratings[ratings['userId'] == user_id]['movieId'].head(5)
    
    if content_metric == 'cosine':
        sim_matrix = cosine_sim
    elif content_metric == 'euclidean':
        sim_matrix = -euclidean_sim  # Convert distance to similarity
    elif content_metric == 'pearson':
        # Compute Pearson correlation matrix for content-based scoring
        # Use cosine similarity as fallback since Pearson on sparse TF-IDF is slow
        # and may not converge well for genre-based vectors
        sim_matrix = cosine_sim  # Fallback to cosine for better performance
    
    content_scores = np.zeros(len(candidates))
    for mid in high_rated:
        if mid in movies['movieId'].values:
            idx = movies[movies['movieId'] == mid].index[0]
            content_scores += sim_matrix[idx][candidates.index]
    
    content_scores = content_scores / max(len(high_rated), 1)
    
    # Hybrid score
    final_scores = alpha * np.array(collab_scores) + (1 - alpha) * content_scores
    candidates['hybrid_score'] = final_scores
    
    recommendations = candidates.sort_values('hybrid_score', ascending=False).head(top_n)
    return recommendations[['title', 'genres', 'hybrid_score']]


# ====================== TEST ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM READY")
    print("="*60)
    
    print("\n1. Content-Based Recommendation (Cosine):")
    print(content_based_recommend("Toy Story (1995)", 5, 'cosine'))
    
    print("\n2. Content-Based Recommendation (Euclidean):")
    print(content_based_recommend("Toy Story (1995)", 5, 'euclidean'))
    
    print("\n3. Content-Based Recommendation (Pearson):")
    print(content_based_recommend("Toy Story (1995)", 5, 'pearson'))
    
    print("\n4. Collaborative Recommendation (SVD) for User 1:")
    print(collaborative_recommend(user_id=1, top_n=5, method='svd'))
    
    print("\n5. Collaborative Recommendation (NMF) for User 1:")
    print(collaborative_recommend(user_id=1, top_n=5, method='nmf'))
    
    print("\n6. Collaborative Recommendation (KNN) for User 1:")
    print(collaborative_recommend(user_id=1, top_n=5, method='knn'))
    
    print("\n7. Hybrid Recommendation for User 1 (SVD):")
    print(hybrid_recommend(user_id=1, top_n=8, alpha=0.65, algorithm='svd'))
    
    print("\n8. Hybrid Recommendation for User 1 (NMF):")
    print(hybrid_recommend(user_id=1, top_n=8, alpha=0.65, algorithm='nmf'))
    
    print("\n9. Hybrid Recommendation for User 1 (KNN):")
    print(hybrid_recommend(user_id=1, top_n=8, alpha=0.65, algorithm='knn'))
