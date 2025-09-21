from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import joblib
from data import load_data, split_data

# def fit_model(X_train, y_train):
#     """
#     Train a Decision Tree Classifier and save the model to a file.
#     Args:
#         X_train (numpy.ndarray): Training features.
#         y_train (numpy.ndarray): Training target values.
#     """
#     dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
#     dt_classifier.fit(X_train, y_train)
#     joblib.dump(dt_classifier, "../model/iris_model.pkl")

def fit_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "../model/diabetes_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
