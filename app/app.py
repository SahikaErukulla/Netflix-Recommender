import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.title("🎬 Netflix Movie Recommendation System")

# Dummy movie data
movies = pd.DataFrame({
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Dark Knight'],
    'description': [
        'dream hacker thriller sci-fi',
        'virtual reality hacker AI',
        'space black hole future exploration',
        'batman joker action hero'
    ]
})

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(movies['description'])
cosine_sim = cosine_similarity(features, features)

movie_titles = movies['title'].tolist()
selection = st.selectbox("Choose a movie:", movie_titles)

if selection:
    idx = movie_titles.index(selection)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    st.subheader("Recommended Movies:")
    for i, score in sim_scores:
        st.write(f"- {movie_titles[i]} (score: {score:.2f})")
