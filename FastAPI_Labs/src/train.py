from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    # dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    # dt_classifier.fit(X_train, y_train)
    model.fit(X_train, y_train)
    # joblib.dump(dt_classifier, "../model/iris_model.pkl")
    joblib.dump(model, "../model/rf_iris_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
