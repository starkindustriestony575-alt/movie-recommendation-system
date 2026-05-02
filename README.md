# 🎬 Movie Recommendation System

A web-based movie recommendation application that uses three different approaches:
- **Content-Based Filtering**: Recommends movies similar to one you liked based on genres
- **Collaborative Filtering**: Recommends movies based on what similar users liked (SVD-based)
- **Hybrid Model** (Recommended): Combines both approaches for better personalization

## 🚀 Live Demo

**GitHub Repository**: https://github.com/starkindustriestony575-alt/movie-recommendation-system

**Streamlit Community Cloud**: (Deploy using instructions below)

## 🛠️ Tech Stack

- **Python 3.14+**
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
    └── movies.pkl          # Cached movie dataframe
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## 📊 How It Works

### 1. Content-Based Filtering
- Uses **TF-IDF** to convert movie genres into numerical vectors
- Calculates **Cosine Similarity** between movies
- Recommends movies with highest similarity scores

### 2. Collaborative Filtering
- Creates a **User-Item Matrix** from ratings data
- Applies **SVD** (Singular Value Decomposition) with 50 latent factors
- Predicts ratings for unrated movies
- Recommends highest-predicted movies

### 3. Hybrid Model (Recommended)
- Combines collaborative scores (α) and content-based scores (1-α)
- Default α = 0.65 (65% collaborative, 35% content-based)
- Provides personalized recommendations

## 🎯 Usage

1. **Select Recommendation Type** from sidebar
2. **Content-Based**: Enter a movie title you liked
3. **Collaborative**: Enter a User ID
4. **Hybrid**: Enter User ID and adjust the balance slider
5. Click **Get Recommendations** to see results


MIT License

## 👤 Sarvagya Gupta

Your Name - [Little Stark](https://github.com/starkindustriestony575-alt)

