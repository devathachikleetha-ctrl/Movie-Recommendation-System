import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎬 Movie Recommender")

st.sidebar.write(
    "This application recommends movies "
    "based on genre similarity using Machine Learning."
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Technology")

st.sidebar.write("🐍 Python")
st.sidebar.write("🐼 Pandas")
st.sidebar.write("🤖 Scikit-learn")
st.sidebar.write("🌐 Streamlit")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset")

st.sidebar.write("MovieLens Dataset")
st.sidebar.write("Number of Movies: 9742")

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favorite movie "
    "using Machine Learning."
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

movies = pd.read_csv("ml-latest-small/movies.csv")
links = pd.read_csv("ml-latest-small/links.csv")

# --------------------------------------------------
# MERGE DATA
# --------------------------------------------------

movies = movies.merge(
    links,
    on="movieId"
)

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

movies["genres"] = movies["genres"].fillna("")

movies["genres"] = movies["genres"].str.replace(
    "|",
    " ",
    regex=False
)

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)

# --------------------------------------------------
# NORMALIZE
# --------------------------------------------------

tfidf_normalized = normalize(
    tfidf_matrix
)

# --------------------------------------------------
# MOVIE POSTER FUNCTION
# --------------------------------------------------

def get_poster_url(imdb_id):

    try:

        imdb_id = str(int(imdb_id)).zfill(7)

        imdb_id = "tt" + imdb_id

        return (
            f"https://images.metahub.space/"
            f"poster/small/{imdb_id}/img"
        )

    except:

        return None


# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

st.subheader("🎥 Select a Movie")

movie_list = movies["title"].tolist()

selected_movie = st.selectbox(
    "Choose your favorite movie:",
    movie_list
)

# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button("🎯 Recommend Movies"):

    movie_index = movies[
        movies["title"] == selected_movie
    ].index[0]

    # Get selected movie vector
    movie_vector = tfidf_normalized[
        movie_index
    ]

    # Calculate similarity
    similarity_scores = (
        tfidf_normalized.dot(
            movie_vector.T
        )
    )

    # Convert to array
    similarity_scores = (
        similarity_scores
        .toarray()
        .flatten()
    )

    # Create index-score pairs
    similar_movies = list(
        enumerate(
            similarity_scores
        )
    )

    # Sort by similarity
    similar_movies = sorted(
        similar_movies,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie
    similar_movies = similar_movies[1:6]

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.success(
        f"Recommendations for: {selected_movie}"
    )

    st.subheader(
        "🍿 Top 5 Recommended Movies"
    )

    for i, (index, score) in enumerate(
        similar_movies,
        start=1
    ):

        movie_name = movies.iloc[
            index
        ]["title"]

        imdb_id = movies.iloc[
            index
        ]["imdbId"]

        poster_url = get_poster_url(
            imdb_id
        )

        col1, col2 = st.columns(
            [1, 3]
        )

        with col1:

            if poster_url:

                st.image(
                    poster_url,
                    width=150
                )

        with col2:

            st.subheader(
                f"{i}. {movie_name}"
            )

            st.write(
                f"⭐ Similarity Score: "
                f"**{score:.2f}**"
            )

        st.divider()

else:

    st.info(
        "👆 Select a movie above and click "
        "'Recommend Movies' to get recommendations."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "🎬 Movie Recommendation System | "
    "Python + Pandas + Scikit-learn + Streamlit"
)