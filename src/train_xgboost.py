from xgboost import XGBRegressor
from model_utils import load_data, evaluate_model, save_result

X_train, X_test, y_train, y_test = load_data()

# n_estimators = 500, meaning 500 trees
# learning_rate represents how much each new tree affects final prediction
    # smaller learning rate = slower learning, but uses more trees
    # bigger learning rate = faster learning, but can overfit often
# max_depth controls how deep each tree can grow
    # setting a value of how many questions it can ask
    # ex. 
    # Question 1: Is danceability > 0.7?
        # yes →
        # Question 2: Is energy > 0.6?
            # yes →
                # Question 3: Is tempo > 120?
                    # yes →
                        # Question 4: Is explicit?
                            # yes → predict popularity around 78
                            # no  → predict popularity around 72
# subsample = 0.8 means that each tree trains 80% of rows from training data
# colsample_bytree means that each tree uses 80% of features/columns
    # prevents all trees from becoming too similar

model = XGBRegressor(
    n_estimators = 500, 
    learning_rate = 0.03,
    max_depth = 4,
    subsample = 0.8, 
    colsample_bytree = 0.8,
    random_state = 30
)

mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)

save_result("XGBRegressor", mae, r2)

print(f"XGBRegressor - MAE: {mae}, R2 Score: {r2}")