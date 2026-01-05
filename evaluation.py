import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

def plot_confusion_matrix(model, X_test, y_test):
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix - Phishing Detection")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - Phishing Detection")
    plt.tight_layout()
    plt.show()
