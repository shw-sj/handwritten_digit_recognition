from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_classification(y_true, y_pred, average="macro"):
    """
    计算常见分类指标
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    cm = confusion_matrix(y_true, y_pred)

    print("=== Classification Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f} ({average})")
    print(f"Recall   : {recall:.4f} ({average})")
    print(f"F1-score : {f1:.4f} ({average})")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }