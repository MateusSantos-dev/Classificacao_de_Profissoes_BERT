import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def detailed_report(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()  # formato serializável
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "report": report,
        "confusion_matrix": matrix,
        "accuracy": accuracy
    }
