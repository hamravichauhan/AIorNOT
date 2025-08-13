from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

DATA_PATH = Path("data/ai_human_content_detection_dataset.csv")  # point to your code dataset if separate
TEXT_COL = "text_content"
LABEL_COL = "label"

MODEL_PATH = Path("models/code_tfidfchar_logreg.joblib")
REPORT_DIR = Path("models/eval_code")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def tune_threshold_by_f1(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (p >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1

def tune_threshold_for_precision(y, p, target=0.90):
    best = (0.5, 0.0, 0.0, 0.0)
    for t in np.linspace(0.05, 0.99, 191):
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if prec >= target:
            best = (float(t), float(prec), float(rec), float(f1))
    return best

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train_code.py` first.")

    pipe = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    y = df[LABEL_COL].astype(int)
    X = df[[TEXT_COL]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    proba = pipe.predict_proba(X_test)[:, 1]
    pred05 = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred05)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred05, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba)

    # Threshold tuning
    best_t, best_f1 = tune_threshold_by_f1(y_test, proba)
    pred_f1 = (proba >= best_t).astype(int)
    prec_f1, rec_f1, f1_f1, _ = precision_recall_fscore_support(y_test, pred_f1, average="binary", zero_division=0)

    t_hp, prec_hp, rec_hp, f1_hp = tune_threshold_for_precision(y_test, proba, target=0.90)

    # ROC plot — save directly to file
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Code Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = REPORT_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)

    # Confusions + reports
    cm05 = confusion_matrix(y_test, pred05)
    pd.DataFrame(cm05, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(REPORT_DIR / "confusion_0.5.csv")

    rep05 = pd.DataFrame(classification_report(y_test, pred05, output_dict=True, zero_division=0)).T
    rep05.to_csv(REPORT_DIR / "classification_report_0.5.csv")

    # Summary JSON
    out = {
        "base_metrics": {
            "threshold": 0.5,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
        },
        "tuned": {
            "best_f1_threshold": round(float(best_t), 3),
            "best_f1_value": round(float(best_f1), 4),
            "best_f1_precision": round(float(prec_f1), 4),
            "best_f1_recall": round(float(rec_f1), 4),
            "high_precision_threshold_0.90": round(float(t_hp), 3),
            "hp_precision": round(float(prec_hp), 4),
            "hp_recall": round(float(rec_hp), 4),
            "hp_f1": round(float(f1_hp), 4),
        },
        "artifacts": {
            "roc_curve_png": str(roc_path),
            "cm_0.5_csv": str(REPORT_DIR / "confusion_0.5.csv"),
            "report_0.5_csv": str(REPORT_DIR / "classification_report_0.5.csv"),
        }
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(out, indent=2))

    print("\n=== Code Evaluation Summary ===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
