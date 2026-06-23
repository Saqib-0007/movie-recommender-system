import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Load Data
# ----------------------------
movies = pickle.load(
    open("movies.pkl", "rb")
)

vectors = pickle.load(
    open("vectors.pkl", "rb")
)

# ----------------------------
# Constants
# ----------------------------
BASE_URL = "https://image.tmdb.org/t/p/w500"

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(movie_name, top_n=5):

    matching_movies = movies[
        movies['title'].str.lower() == movie_name.lower()
    ]

    if matching_movies.empty:
        return {
            "status": "error",
            "message": f"Movie '{movie_name}' not found."
        }

    try:
        idx = matching_movies.index[0]

        distances = cosine_similarity(
            vectors[idx],
            vectors
        ).flatten()

        scored_movies = []

        for i, similarity in enumerate(distances):

            if i == idx:
                continue

            movie_data = movies.iloc[i]

            imdb_score = float(movie_data["imdb_rating"]) / 10

            final_score = (
                0.9 * similarity +
                0.1 * imdb_score
            )

            scored_movies.append(
                (i, final_score)
            )

        similar_movies = sorted(
            scored_movies,
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        recommendations = []

        for movie in similar_movies:

            movie_data = movies.iloc[movie[0]]

            recommendations.append({
                "title": movie_data["title"],
                "rating": float(movie_data["imdb_rating"]),
                "poster": movie_data["poster_path"]
            })

        return {
            "status": "success",
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ----------------------------
# UI
# ----------------------------
st.title("🎬 Movie Recommender")
st.write("Find movies similar to your favorite movies")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

# ----------------------------
# Recommendation Button
# ----------------------------
if st.button("Recommend"):

    with st.spinner("Finding recommendations..."):

        result = recommend(selected_movie)

    if result["status"] == "success":

        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        cols = st.columns(5)

        for col, movie in zip(
            cols,
            result["recommendations"]
        ):

            with col:

                st.image(
                    BASE_URL + movie["poster"]
                )

                st.write(
                    movie["title"]
                )

                st.write(
                    f"⭐ {movie['rating']}"
                )

    else:
        st.error(
            result["message"]
        )