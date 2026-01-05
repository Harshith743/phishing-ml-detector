from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from decision_tree import load_dataset


if __name__ == "__main__":
    print("🔐 Phishing Website Detection using Logistic Regression")

    X_train, y_train, X_test, y_test = load_dataset()
    print("📂 Dataset loaded successfully")

    model = LogisticRegression(max_iter=200)
    print("📈 Logistic Regression model created")

    model.fit(X_train, y_train)
    print("🤖 Model training complete")

    predictions = model.predict(X_test)

    accuracy = 100 * accuracy_score(y_test, predictions)
    print(f"✅ Model accuracy on test data: {accuracy:.2f}%")
