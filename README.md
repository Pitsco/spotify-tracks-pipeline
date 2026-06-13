# Spotify Tracks ETL and Machine Learning Pipeline

This project uses a Spotify Tracks dataset from Kaggle to build an end-to-end data project. It includes an ETL pipeline, exploratory data analysis, machine learning model training, model comparison, and an interactive Streamlit app.

## Project Overview

The goal of this project is to:

- Extract raw Spotify track data from a CSV file
- Clean and transform the dataset
- Save a processed version of the data
- Explore patterns in Spotify audio features
- Train machine learning models to predict track popularity
- Compare Linear Regression, Random Forest, and XGBoost models
- Build an interactive app to test popularity predictions

## Dataset

Dataset used: [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

The dataset includes information such as:

- Track name
- Artists
- Album name
- Popularity
- Duration
- Explicit status
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Track genre

Resulted Outcome:

<img width="602" height="286" alt="Screenshot 2026-06-12 at 1 46 56 PM" src="https://github.com/user-attachments/assets/86e657fe-dbaa-458b-81b9-f8f70d09ea7d" />

## What I've learned from all 3 models
Linear Regression is a model where two variables, x and y, create a linear expression that predicts future outcomes. In this model, I condensed the independent variables (columns) into a list used to predict the dependent variable, popularity.

In this experiment, the relationships between the independent variables and popularity (the dependent variable) appear to be highly nonlinear. 

I was thinking about it, and there are many examples where the values can be:
- High energy might help pop songs, but not acoustic songs
- Tempo might matter only within certain ranges
- Instrumentalness might strongly lower popularity for some genres but not others
- Danceability and energy together may matter more than either one alone

Because Linear Regression mainly captures straight-line relationships between each feature and the target, it may struggle to model these interactions and nonlinear patterns. This helps explain why more flexible models, such as Random Forest, performed better in this experiment.

