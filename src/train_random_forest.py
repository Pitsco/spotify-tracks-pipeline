from sklearn.ensemble import RandomForestRegressor
from model_utils import load_data, evaluate_model, save_result

#loading data 
X_train, X_test, y_train, y_test = load_data()

# n_estimators means that the randomforest model will use 100 decision trees

model = RandomForestRegressor(
    n_estimators=100,
    random_state=3
)

mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)

save_result("Random Forest", mae, r2)

print(f"Random Forest - MAE: {mae}, R2 Score: {r2}")