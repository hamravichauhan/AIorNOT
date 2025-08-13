from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

from utils import Config, load_raw_dataframe, basic_clean, get_xy, train_test_split_xy

OUTPUT_DIR = Path("models")
MODEL_PATH = OUTPUT_DIR / "baseline_tfidf_logreg.joblib"
REPORT_DIR = OUTPUT_DIR / "eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def tune_threshold_by_f1(y_true, proba):
    """Return threshold that maximizes F1 on the eval set."""
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 37):  # step ~0.025
        pred = (proba >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1

def tune_threshold_for_target_precision(y_true, proba, target_precision=0.9):
    """
    Find the *highest* threshold achieving at least `target_precision`.
    Returns (threshold, precision, recall, f1).
    """
    best = (0.5, 0.0, 0.0, 0.0)
    for t in np.linspace(0.05, 0.99, 191):
        pred = (proba >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if prec >= target_precision:
            best = (float(t), float(prec), float(rec), float(f1))
    return best

def main():
    # 1) Load model + data split
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")

    pipe = joblib.load(MODEL_PATH)

    cfg = Config()
    df = load_raw_dataframe(cfg)
    df = basic_clean(df, cfg)
    X, y = get_xy(df, cfg)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, cfg)

    # 2) Eval with current 0.5 threshold
    proba = pipe.predict_proba(X_test)[:, 1]
    pred05 = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred05)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred05, average='binary', zero_division=0)
    auc = roc_auc_score(y_test, proba)

    base_metrics = {
        "threshold": 0.5,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4)
    }

    # 3) Threshold tuning
    best_t, best_f1 = tune_threshold_by_f1(y_test, proba)
    pred_f1 = (proba >= best_t).astype(int)
    prec_f1, rec_f1, f1_f1, _ = precision_recall_fscore_support(y_test, pred_f1, average='binary', zero_division=0)

    # Example: if you need high precision (avoid false accusations)
    t_prec, prec_hp, rec_hp, f1_hp = tune_threshold_for_target_precision(y_test, proba, target_precision=0.9)

    tuned = {
        "best_f1_threshold": round(best_t, 3),
        "best_f1_value": round(float(best_f1), 4),
        "best_f1_precision": round(float(prec_f1), 4),
        "best_f1_recall": round(float(rec_f1), 4),

        "high_precision_threshold_0.90": round(float(t_prec), 3),
        "hp_precision": round(float(prec_hp), 4),
        "hp_recall": round(float(rec_hp), 4),
        "hp_f1": round(float(f1_hp), 4),
    }

    # 4) Save ROC curve
    fpr, tpr, thr = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Baseline TF-IDF + Logistic Regression")
    plt.legend(loc="lower right")
    roc_path = REPORT_DIR / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # 5) Save confusion matrices (0.5 and best-F1)
    cm05 = confusion_matrix(y_test, pred05)
    cm_f1 = confusion_matrix(y_test, pred_f1)
    pd.DataFrame(cm05, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(REPORT_DIR / "confusion_0.5.csv")
    pd.DataFrame(cm_f1, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(REPORT_DIR / "confusion_bestF1.csv")

    # 6) Save classification reports
    rep05 = pd.DataFrame(classification_report(y_test, pred05, output_dict=True, zero_division=0)).T
    rep_f1 = pd.DataFrame(classification_report(y_test, pred_f1, output_dict=True, zero_division=0)).T
    rep05.to_csv(REPORT_DIR / "classification_report_0.5.csv")
    rep_f1.to_csv(REPORT_DIR / "classification_report_bestF1.csv")

    # 7) Token explanations (only for TF-IDF + Linear model)
    #    If pipeline changes later, guard for attribute presence.
    top_path_ai = REPORT_DIR / "top_tokens_AI.csv"
    top_path_human = REPORT_DIR / "top_tokens_HUMAN.csv"
    try:
        vec = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        feature_names = np.array(vec.get_feature_names_out())
        coefs = clf.coef_.ravel()

        top_ai_idx = np.argsort(coefs)[-40:][::-1]
        top_human_idx = np.argsort(coefs)[:40]

        pd.DataFrame({
            "token": feature_names[top_ai_idx],
            "weight": coefs[top_ai_idx]
        }).to_csv(top_path_ai, index=False)

        pd.DataFrame({
            "token": feature_names[top_human_idx],
            "weight": coefs[top_human_idx]
        }).to_csv(top_path_human, index=False)
    except Exception as e:
        print("Skipping token explanations:", e)

    # 8) Save metrics JSON
    out = {
        "base_metrics": base_metrics,
        "tuned": tuned,
        "artifacts": {
            "roc_curve_png": str(roc_path),
            "cm_0.5_csv": str(REPORT_DIR / "confusion_0.5.csv"),
            "cm_bestF1_csv": str(REPORT_DIR / "confusion_bestF1.csv"),
            "report_0.5_csv": str(REPORT_DIR / "classification_report_0.5.csv"),
            "report_bestF1_csv": str(REPORT_DIR / "classification_report_bestF1.csv"),
            "top_tokens_AI_csv": str(top_path_ai),
            "top_tokens_HUMAN_csv": str(top_path_human),
        }
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(out, indent=2))

    print("\n=== Evaluation Summary ===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
