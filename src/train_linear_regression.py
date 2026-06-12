from sklearn.linear_model import LinearRegression
from model_utils import load_data, evaluate_model, save_result

# loading both test and training data using load data function
X_train, X_test, y_train, y_test = load_data()

# loading linear regression model and creates it as an object
# doesn't train it!
model = LinearRegression()

# train model on the training data, and testing it on the testing data
# returns mae and r2 score
mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
save_result("Linear Regression", mae, r2)

print(f"Linear Regression - MAE: {mae}, R2 Score: {r2}")