#importing python's operating system module
import os
import pandas as pd

# imports train_test_spilt from scikit-learn
# training data -> used to train the model
# testing data -> used to evaluate the model
from sklearn.model_selection import train_test_split

# mean_absolute_error measures how far off your predictions are on average
# r2_score measures how well your model explains the target value
    # r2 score close to 10 is usually good
    # score of 0 means its not doing better than guessing the average
from sklearn.metrics import mean_absolute_error, r2_score

# columns that the model will use to make predictions
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_minutes",
    "explicit"
]

def load_data():
    df = pd.read_csv("../data/processed/spotify_tracks_cleaned.csv")

    # converts explicit column into integers
    # therefore False = 0 and True = 1
    df["explicit"] = df["explicit"].astype(int)

    # X represents inputs
    X = df[FEATURES]

    # y contains value that we want to predict
    y = df["popularity"]

    # function from sckit-learn
        # X is the training data
        # y is the testing data to evaluate the model
    # test_size is the % of data used for testing 
    # and the rest is used for training
    # random_state is used so that we get same split of data every time
    # therefore the data is not random everytime we run the code 

    return train_test_split(
        X,
        y, 
        test_size = 0.2,
        random_state = 42
    )

# takes in 5 inputs: model, X_train, X_test, y_train, y_test
# model = machine learning model
# X_train = training song features
# X_test = testing song features
# y_train = actual popularity values of training songs
# y_test = actualy popularity values of testing songs
def evalute_model(model, X_train, X_test, y_train, y_test):
    # trains the model
        # model should be learning that high danceability + high energy = high popularity
        # songs with certain tempos have certain popularity trends
    model.fit(X_train, y_train)

    # makes predictions based on testing data (data that hasn't been seen)
    predictions = model.predict(X_test)

    # Tells u how wrong the model is on average
    # For example:
        # Actual populartiy = 90
        # Predicted popularity = 80
        # Absolute error = 10
    # lower the mae, the better
    mae = mean_absolute_error(y_test, predictions)

    # calculates r^2 score, tells u how well the model explains the real popularity values
    # close to 1 is good, close to 0 is bad, below 0 is really bad

    r2 = r2_score(y_test, predictions)

    # we use y_test and predictions because y_test is actual popularity
    # and predicitions tell use the model's predicted popularity
    # so we compare both

    return mae, r2

def save_result(model_name, mae, r2):
    # creates a folder called model_results
    # "exist_ok = True" and if it exists, don't crash
        # if it were to exist, Python would give an error since
        # basically saying if the folder exists, use it
    os.makedirs("data/processed/model_results", exist_ok=True)

    # creating a pandas dataframe called result 
    # 3 columns, with their corresponding values
    result = pd.DataFrame({
        "Model": model_name,
        "MAE": mae,
        "R2 Score": r2
    })

    # lower casing and replacing spaces 
    file_name = model_name.lower().replace(" ", "_")

    # save the df file to a csv and removes the index column
        # index is not saved because we dont want it in our data
    result.to_csv(f"data/processed/model_results/{file_name}.csv", index=False)

