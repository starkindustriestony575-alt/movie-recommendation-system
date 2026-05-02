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

## 🚀 Deploy to Streamlit Cloud (Easiest)

1. **Push to GitHub** (see below)
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app" → Select your repo → Select branch → Main file path: `app.py`
5. Click "Deploy!"

Your app will be live at `https://yourusername-movie-recommendation-system.streamlit.app`

## 📤 Push to GitHub (Manual Steps)

Since GitHub CLI isn't installed, follow these steps:

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Repository name: `movie-recommendation-system`
   - Make it **Public**
   - Don't add README (we already have one)
   - Click "Create repository"

2. **Push your code** (copy-paste these commands in terminal):
   ```
   echo "# Movie Recommendation System" >> README.md
   git remote add origin https://github.com/YOURUSERNAME/movie-recommendation-system.git
   git branch -M main
   git push -u origin main
   ```
   (Replace `YOURUSERNAME` with your GitHub username)

3. Your code is now on GitHub! 🎉

## 🎯 Usage

1. **Select Recommendation Type** from sidebar
2. **Content-Based**: Enter a movie title you liked
3. **Collaborative**: Enter a User ID
4. **Hybrid**: Enter User ID and adjust the balance slider
5. Click **Get Recommendations** to see results


MIT License

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)
