import streamlit as st
import pandas as pd
from main import (
    content_based_recommend, 
    collaborative_recommend, 
    hybrid_recommend, 
    movies,
    MAX_USER_ID
)


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎥 AI Movie Recommendation System")
st.markdown("**Content-based + Collaborative Filtering + Hybrid Model**")

# Sidebar
st.sidebar.header("Choose Recommendation Type")

option = st.sidebar.radio(
    "Select Method:",
    ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Model (Recommended)"]
)

# ====================== CONTENT-BASED ======================
if option == "Content-Based Filtering":
    st.subheader("🎬 Content-Based Recommendation")
    st.write("Find movies similar to one you like based on genres.")
    
    movie_title = st.text_input("Enter a movie title you enjoyed:", 
                               value="Dark Knight The (2008)")
    
    num_recs = st.slider("Number of recommendations", 5, 15, 10)
    
    if st.button("Get Similar Movies", type="primary"):
        with st.spinner("Finding similar movies..."):
            recommendations = content_based_recommend(movie_title, num_recs)
            
            if "Error" in recommendations.columns:
                st.error(recommendations.iloc[0, 0])
            else:
                st.success(f"Movies similar to **{movie_title}**")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)

# ====================== COLLABORATIVE ======================
elif option == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering")
    st.write("Recommendations based on what similar users liked.")
    
    user_id = st.number_input("Enter User ID", min_value=1, max_value=int(MAX_USER_ID), value=1)

    num_recs = st.slider("Number of recommendations", 5, 15, 10, key="collab_slider")
    
    if st.button("Get Recommendations for User", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = collaborative_recommend(user_id, num_recs)
            
            if "Error" in recommendations.columns:
                st.error(recommendations.iloc[0, 0])
            else:
                st.success(f"Top recommendations for User **{user_id}**")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)

# ====================== HYBRID (Best) ======================
else:
    st.subheader("🔥 Hybrid Model (Recommended)")
    st.write("Combines Collaborative + Content-based for better personalization.")
    
    user_id = st.number_input("Enter User ID", min_value=1, max_value=int(MAX_USER_ID), value=1, key="hybrid_user")

    num_recs = st.slider("Number of recommendations", 5, 15, 10, key="hybrid_slider")
    alpha = st.slider("Balance (Collaborative vs Content)", 0.0, 1.0, 0.65, 0.05)
    
    if st.button("Get Personalized Recommendations", type="primary"):
        with st.spinner("Calculating hybrid recommendations..."):
            recommendations = hybrid_recommend(user_id, num_recs, alpha)
            
            if "Error" in recommendations.columns:
                st.error(recommendations.iloc[0, 0])
            else:
                st.success(f"Hybrid Recommendations for User **{user_id}**")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
                
                st.caption("Higher `hybrid_score` = Better recommendation")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Dataset: MovieLens ml-latest-small\n"
    "Techniques: Cosine Similarity + SVD Matrix Factorization"
)

st.caption("Built in VS Code with Python • Content-based + Collaborative + Hybrid")
