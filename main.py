from decision_tree import load_dataset, train_decision_tree
from logistic_regression import LogisticRegression
import joblib

if __name__ == "__main__":
    print("🚀 Running phishing detection project")

    X_train, y_train, X_test, y_test = load_dataset()

    model = train_decision_tree(X_train, y_train)

    joblib.dump(model, "phishing_model.pkl")
    print("💾 Model saved as phishing_model.pkl")
