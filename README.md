# 🎬 Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8C0495?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)

[![Stars](https://img.shields.io/github/stars/starkindustriestony575-alt/movie-recommendation-system?style=flat&color=yellow)](https://github.com/starkindustriestony575-alt/movie-recommendation-system/stargazers)
[![Forks](https://img.shields.io/github/forks/starkindustriestony575-alt/movie-recommendation-system?style=flat&color=green)](https://github.com/starkindustriestony575-alt/movie-recommendation-system/fork)
[![License](https://img.shields.io/github/license/starkindustriestony575-alt/movie-recommendation-system?style=flat&color=blue)](https://github.com/starkindustriestony575-alt/movie-recommendation-system/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/starkindustriestony575-alt/movie-recommendation-system?style=flat&color=orange)](https://github.com/starkindustriestony575-alt/movie-recommendation-system/commits/main)

---

## ✨ Overview

A modern, web-based movie recommendation application built with **Python** and **Streamlit**, featuring three different industry-standard recommendation approaches to deliver personalized movie suggestions.

### 🎯 Key Features

| Feature | Description |
| --------- | ------------- |
| 🧠 **Content-Based Filtering** | Recommends movies similar to your favorites using genre analysis with TF-IDF & Cosine Similarity |
| 👥 **Collaborative Filtering** | Leverages community preferences using SVD/NMF/KNN matrix factorization |
| 🔄 **Hybrid Model** | Combines both approaches for superior personalization (default 65% collaborative / 35% content) |
| ⚡ **Pre-computed Models** | Near-instant recommendations with cached similarity matrices |
| 🎨 **Interactive UI** | Beautiful Streamlit interface with real-time recommendations |

---

## 🚀 Live Demo

> **Try it now!** No installation required.

| Platform | Link |
| -------- | ---- |
| 🌐 **Streamlit Cloud** | [movie-recommendation-system.streamlit.app](https://movie-recommendation-system-vsdarx6jwnpi8ft5wyqjzv.streamlit.app/) |
| 💻 **GitHub Repository** | [github.com/starkindustriestony575-alt/movie-recommendation-system](https://github.com/starkindustriestony575-alt/movie-recommendation-system) |

---

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.14+** — Programming Language
- **Streamlit** — Web UI Framework
- **Pandas** — Data manipulation & analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — ML algorithms (TF-IDF, Cosine Similarity, KNN)
- **SciPy** — SVD/NMF matrix factorization

### Data Source

- **MovieLens ml-latest-small** by GroupLens Research
  - 📽️ 9,742 movies
  - 👤 100,004 ratings from 610 users

---

## 📁 Project Structure

```text
Movie recommendation-system/
├── 📂 app.py                  # Streamlit web application (UI layer)
├── 📂 main.py                 # Core recommendation algorithms (business logic)
├── 📂 requirements.txt        # Python dependencies
├── 📂 README.md               # This file
├── 📂 LICENSE                 # MIT License
├── 📂 Data/
│   ├── 📄 movies.csv          # Movie metadata (ID, title, genres)
│   └── 📄 ratings.csv         # User ratings (userId, movieId, rating, timestamp)
└── 📂 models/
    ├── 📄 cosine_sim.npy      # Pre-computed Cosine similarity matrix
    ├── 📄 euclidean_sim.npy  # Pre-computed Euclidean similarity matrix
    ├── 📄 tfidf_normalized.npy # Normalized TF-IDF vectors
    └── 📄 movies.pkl         # Cached movie dataframe
```

---

## 📊 How It Works

### 1️⃣ Content-Based Filtering

> *"Show me more movies like what I enjoyed"*

| Component | Details |
| ----------- | --------- |
| **Vectorization** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Similarity Metrics** | Cosine Similarity, Euclidean Distance, Pearson Correlation |
| **Input** | Movie title (e.g., "The Dark Knight (2008)") |
| **Output** | Top-N similar movies ranked by similarity score |

**Algorithm Flow:**

```text
Movie Genres → TF-IDF Vectorization → Similarity Calculation → Ranked Recommendations
```

### 2️⃣ Collaborative Filtering

> *"What did users like me also enjoy?"*

| Component | Details |
| ----------- | --------- |
| **Matrix** | User-Item Ratings Matrix |
| **Algorithms** | SVD (50 latent factors), NMF, KNN |
| **Input** | User ID (1-610) |
| **Output** | Predicted ratings for unrated movies |

**Algorithm Flow:**

```text
User Ratings → User-Item Matrix → SVD/NMF/KNN → Rating Prediction → Ranked Recommendations
```

### 3️⃣ Hybrid Model (Recommended ⚡)

> *"The best of both worlds"*

| Component | Details |
| ----------- | --------- |
| **Approach** | Weighted combination of Content + Collaborative |
| **Default Balance** | α = 0.65 (65% collaborative, 35% content-based) |
| **Customizable** | Adjustable α slider (0.0 - 1.0) |
| **Input** | User ID + optional movie selection |
| **Output** | Highly personalized recommendations |

**Algorithm Flow:**

```text
Content Scores ─┬─→ Weighted Combine → Final Ranking → Personalized Recommendations
Collaborative ┘
```

---

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

### Step 4: Open in Browser

Navigate to: **<http://localhost:8501>**

---

## 📖 Usage Guide

### 🎬 Content-Based Filtering

1. Select **Content-Based Filtering** from the sidebar
2. Choose a similarity metric:
   - 🔹 **Cosine** (default) — Measures angle between vectors
   - 🔹 **Euclidean** — Measures direct distance
   - 🔹 **Pearson** — Measures linear correlation
3. Enter a movie title you enjoyed
   - *Example:* `The Dark Knight (2008)`
4. Adjust the **Number of Recommendations** slider
5. Click **Get Similar Movies**

### 👤 Collaborative Filtering

1. Select **Collaborative Filtering** from the sidebar
2. Choose an algorithm:
   - 🔹 **SVD** (default) — Singular Value Decomposition
   - 🔹 **NMF** — Non-negative Matrix Factorization
   - 🔹 **KNN** — K-Nearest Neighbors
3. Enter a **User ID** (1-610)
   - *Example:* `1`
4. Adjust the **Number of Recommendations** slider
5. Click **Get Recommendations for User**

### ⚡ Hybrid Model (Recommended)

1. Select **Hybrid Model (Recommended)** from the sidebar
2. Choose collaboration algorithm:
   - 🔹 **SVD** (default) / NMF / KNN
3. Choose content similarity metric:
   - 🔹 **Cosine** (default) / Euclidean
4. Enter a **User ID** (1-610)
5. Adjust the **Balance** slider (α):
   - Left (0.0) = 100% Content-Based
   - Center (0.5) = 50/50 Balanced
   - Right (1.0) = 100% Collaborative
   - **Default: 0.65** (65% Collaborative, 35% Content)
6. Click **Get Personalized Recommendations**

---

## 🔧 Algorithm Parameters

### Content-Based Filtering

| Parameter | Type | Default | Range | Description |
| ----------- | ------ | --------- | ------- | ------------- |
| `top_n` | int | 10 | 1-50 | Number of recommendations |
| `metric` | str | 'cosine' | cosine/euclidean/pearson | Similarity metric |

### Collaborative Filtering

| Parameter | Type | Default | Range | Description |
| ----------- | ------ | --------- | ------- | ------------- |
| `user_id` | int | 1 | 1-610 | Target user ID |
| `top_n` | int | 10 | 1-50 | Number of recommendations |
| `method` | str | 'svd' | svd/nmf/knn | Matrix factorization method |
| `n_factors` | int | 50 | 10-100 | Latent factors (SVD/NMF only) |

### Hybrid Model

| Parameter | Type | Default | Range | Description |
| ----------- | ------ | --------- | ------- | ------------- |
| `user_id` | int | 1 | 1-610 | Target user ID |
| `top_n` | int | 10 | 1-50 | Number of recommendations |
| `alpha` | float | 0.65 | 0.0-1.0 | Collaborative weight (1-α = content) |
| `content_metric` | str | 'cosine' | cosine/euclidean | Content similarity metric |

---

## 🤝 Contributing

Contributions are welcome! Help us make this project even better.

### How to Contribute

1. 🍴 **Fork** the repository
2. 🌿 **Create** a new branch: `git checkout -b feature/amazing-feature`
3. 💾 **Commit** your changes: `git commit -m 'Add amazing feature'`
4. 📤 **Push** to the branch: `git push origin feature/amazing-feature`
5. 🔃 **Open** a Pull Request

### Contribution Ideas

| Area | Ideas |
| ------ | ------- |
| 🔬 **Algorithms** | Add neural collaborative filtering, autoencoders, graph-based recommendations |
| 📊 **Metrics** | Implement RMSE, MAE, Precision@K, Recall@K evaluation |
| 🎨 **UI/UX** | Add movie posters, trailers, ratings visualization |
| 👤 **Features** | User authentication, watchlists, viewing history |
| 📱 **Platforms** | REST API with FastAPI, Docker containerization |

---

## 📝 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sarvagya Gupta**  
*(Little Stark)*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/starkindustriestony575-alt) [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:starkgupta575@gmail.com)

---

⭐ If you found this project useful, please give it a star!

Happy Movie Watching! 🍿
