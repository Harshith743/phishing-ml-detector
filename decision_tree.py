from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
from evaluation import plot_confusion_matrix, plot_roc_curve


def load_dataset():
    """
    Load dataset from dataset.csv and split into
    training and test sets.
    """
    data = np.genfromtxt("dataset.csv", delimiter=",", dtype=np.int32)

    features = data[:, :-1]
    labels = data[:, -1]

    train_features = features[:2000]
    train_labels = labels[:2000]

    test_features = features[2000:]
    test_labels = labels[2000:]

    return train_features, train_labels, test_features, test_labels


def train_decision_tree(train_X, train_y):
    """Create and train Decision Tree classifier"""
    model = tree.DecisionTreeClassifier(random_state=42)
    model.fit(train_X, train_y)
    return model


if __name__ == "__main__":
    print("🔐 Phishing Website Detection using Decision Tree")

    X_train, y_train, X_test, y_test = load_dataset()
    print("📂 Dataset loaded successfully")

    model = train_decision_tree(X_train, y_train)
    print("🌳 Decision tree trained")

    predictions = model.predict(X_test)

    accuracy = 100 * accuracy_score(y_test, predictions)
    print(f"✅ Model accuracy on test data: {accuracy:.2f}%")

    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    print("📊 Evaluation plots generated")