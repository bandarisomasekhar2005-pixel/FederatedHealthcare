"""
metrics.py
----------
Comprehensive clinical evaluation metrics for cardiac disease prediction.
Computes Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score,
and Area Under the ROC Curve (ROC-AUC).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_clinical_predictions(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes standard healthcare diagnostic metrics.

    Parameters:
      y_true : Ground truth binary labels (0 or 1)
      y_pred : Predicted binary labels (0 or 1)
      y_prob : Predicted probability for positive class (0.0 to 1.0)

    Returns:
      Dictionary of metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / Recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity

    # ROC-AUC calculation
    auc = 0.5
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)

    return {
        "Accuracy": float(acc),
        "Sensitivity (Recall)": float(sens),
        "Specificity": float(spec),
        "Precision": float(prec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }
