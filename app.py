import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Spotify Tracks ETL + ML Dashboard")

st.write("""
This project uses a Spotify tracks dataset to build an ETL pipeline,
explore audio features, and compare machine learning models that predict track popularity.
""")


# Load cleaned data
df = pd.read_csv("data/processed/spotify_tracks_cleaned.csv")


st.header("Dataset Preview")

st.write("Here are the first few rows of the cleaned dataset:")
st.dataframe(df.head())


st.header("Dataset Summary")

st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")


st.header("Top 10 Genres by Average Popularity")

genre_popularity = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
genre_popularity.plot(kind="bar", ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel("Average Popularity")
ax.set_title("Top 10 Genres by Average Popularity")
plt.xticks(rotation=45)
st.pyplot(fig)


st.header("Feature vs Popularity")

feature = st.selectbox(
    "Choose an audio feature:",
    ["danceability", "energy", "loudness", "tempo", "valence", "acousticness"]
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[feature], df["popularity"], alpha=0.3)
ax.set_xlabel(feature)
ax.set_ylabel("Popularity")
ax.set_title(f"{feature} vs Popularity")
st.pyplot(fig)

st.header("Model Comparison")

results = pd.read_csv("data/processed/model_comparison_results.csv")

st.dataframe(results)

best_model = results.sort_values(by="MAE").iloc[0]

st.write(f"Best model based on MAE: **{best_model['Model']}**")
st.write(f"MAE: **{best_model['MAE']:.2f}**")
st.write(f"R² Score: **{best_model['R2 Score']:.2f}**")