from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"=== {model_name} ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["small", "fit", "large"], zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    return {
        "model": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
    }