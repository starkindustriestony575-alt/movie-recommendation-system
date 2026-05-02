# 🎬 Movie Recommendation System

A web-based movie recommendation application that uses three different approaches:
- **Content-Based Filtering**: Recommends movies similar to one you liked based on genres
- **Collaborative Filtering**: Recommends movies based on what similar users liked (SVD-based)
- **Hybrid Model** (Recommended): Combines both approaches for better personalization

## 🚀 Live Demo

**GitHub Repository**: https://github.com/starkindustriestony575-alt/movie-recommendation-system

**Streamlit Community Cloud**: https://movie-recommendation-system-vsdarx6jwnpi8ft5wyqjzv.streamlit.app/

## 🛠️ Tech Stack

- **Python 3.14+** - Programming Language
- **Streamlit** - Web UI Framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - TF-IDF Vectorization & Cosine Similarity
- **SciPy** - SVD (Singular Value Decomposition)

## 📁 Project Structure

```
Movie recommendation-system/
├── app.py                  # Streamlit web application
├── main.py                 # Core recommendation algorithms
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── Data/
│   ├── movies.csv          # Movie metadata
│   └── ratings.csv         # User ratings data
└── models/
    ├── cosine_sim.npy       # Pre-computed cosine similarity matrix
    ├── euclidean_sim.npy   # Pre-computed euclidean similarity matrix
    ├── tfidf_normalized.npy # Normalized TF-IDF vectors
    └── movies.pkl          # Cached movie dataframe
```

## 📊 How It Works

### 1. Content-Based Filtering
- Uses **TF-IDF** to convert movie genres into numerical vectors
- Calculates **Cosine Similarity** between movies
- Recommends movies with highest similarity scores
- Supports multiple metrics: Cosine, Euclidean Distance, Pearson Correlation

### 2. Collaborative Filtering
- Creates a **User-Item Matrix** from ratings data
- Applies **SVD** (Singular Value Decomposition) with 50 latent factors
- Also supports NMF (Non-negative Matrix Factorization) and KNN
- Predicts ratings for unrated movies
- Recommends highest-predicted movies

### 3. Hybrid Model (Recommended)
- Combines collaborative scores (α) and content-based scores (1-α)
- Default α = 0.65 (65% collaborative, 35% content-based)
- Provides personalized recommendations

## 🏃 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/starkindustriestony575-alt/movie-recommendation-system.git
cd movie-recommendation-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

### Step 4: Access the Web App

Open your browser and navigate to:
- **Local**: http://localhost:8501

## 📖 Usage Guide

### Content-Based Filtering

1. Select **Content-Based Filtering** from the sidebar
2. Choose a similarity metric (Cosine, Euclidean, or Pearson)
3. Enter a movie title you enjoyed (e.g., "Toy Story (1995)")
4. Adjust the number of recommendations
5. Click **Get Similar Movies**

**Example Input**: `The Dark Knight (2008)`

### Collaborative Filtering

1. Select **Collaborative Filtering** from the sidebar
2. Choose an algorithm (SVD, NMF, or KNN)
3. Enter a User ID (1-610)
4. Adjust the number of recommendations
5. Click **Get Recommendations for User**

**Example Input**: User ID: `1`

### Hybrid Model

1. Select **Hybrid Model (Recommended)** from the sidebar
2. Choose collaboration algorithm (SVD, NMF, or KNN)
3. Choose content similarity metric (Cosine or Euclidean)
4. Enter a User ID
5. Adjust the balance slider (α) to weight collaborative vs content-based
6. Click **Get Personalized Recommendations**

**Example Input**: User ID: `1`, α = `0.65`

## 📦 Data Source

This project uses the **MovieLens ml-latest-small** dataset:
- **movies.csv**: Contains 9,742 movies with movieId, title, and genres
- **ratings.csv**: Contains 100,004 ratings from 610 users

Dataset Source: https://grouplens.org/datasets/movielens/ml-latest-small/

## 🔧 Algorithm Details

### Content-Based Filtering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | int | 10 | Number of recommendations |
| `metric` | str | 'cosine' | Similarity metric (cosine/euclidean/pearson) |

### Collaborative Filtering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | int | 1 | User ID for recommendations |
| `top_n` | int | 10 | Number of recommendations |
| `method` | str | 'svd' | Algorithm (svd/nmf/knn) |

### Hybrid Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | int | 1 | User ID for recommendations |
| `top_n` | int | 10 | Number of recommendations |
| `alpha` | float | 0.65 | Balance weight (0.0-1.0) |
| `content_metric` | str | 'cosine' | Content similarity metric |

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Ideas

- Add more similarity metrics
- Implement new recommendation algorithms
- Improve the UI/UX
- Add movie posters or trailers
- Implement user authentication
- Add rating prediction accuracy metrics

## 📝 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sarvagya Gupta** (Little Stark)

- GitHub: https://github.com/starkindustriestony575-alt
- Email: starkgupta575@gmail.com

---

⭐ If you found this project useful, please give it a star!

Happy Movie Watching! 🍿
