import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

PIPELINE_ROOT = Path(__file__).resolve().parent

def compute_classification_metrics(y_true, y_pred, y_prob):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)), 
        "accuracy": float(accuracy_score(y_true, y_pred)), 
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)), 
        "precision": float(precision_score(y_true, y_pred, zero_division=0)), 
        "recall": float(recall_score(y_true, y_pred, zero_division=0)), 
        "confusion_matrix": cm.tolist(), 
        "support": {
            "total": int(len(y_true)), 
            "positive": int(np.sum(y_true == 1)), 
            "negative": int(np.sum(y_true == 0))
        }
    }

def save_metrics(metrics_dict: dict, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

def load_metrics(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)

def compare_model_metrics(
    metrics_dir: str | Path | None = None, 
    model_names: list[str] | None = None, 
    output_path: str | Path | None = None
):
    if (metrics_dir is None):
        metrics_dir = PIPELINE_ROOT / "outputs" / "metrics"
    metrics_dir = Path(metrics_dir)
    if (model_names is None):
        model_names = ["lstm", "gru", "tcn"]
    rows = []
    for name in model_names:
        path = metrics_dir / f"{name}_metrics.json"
        if (not path.exists()):
            continue
        m = load_metrics(path)
        rows.append({
            "model": name, 
            "roc_auc": m.get("roc_auc"), 
            "accuracy": m.get("accuracy"), 
            "f1_score": m.get("f1_score"), 
            "precision": m.get("precision"), 
            "recall": m.get("recall")
        })
    if (not rows):
        raise ValueError("No metrics found for the specified models.")

    df = pd.DataFrame(rows).set_index("model")
    # Print the DataFrame with 4 decimal places for the metrics.
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    if (output_path is None):
        output_path = metrics_dir / "model_comparison.csv"
    output_path = Path(output_path)
    df.to_csv(output_path)
    return df