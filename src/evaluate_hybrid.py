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
from sklearn.model_selection import train_test_split

from utils import Config, load_raw_dataframe, basic_clean

MODEL_PATH = Path("models/hybrid_svd_num_logreg.joblib")
REPORT_DIR = Path("models/eval_hybrid")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def select_numeric_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "word_count","character_count","sentence_count","lexical_diversity",
        "avg_sentence_length","avg_word_length","punctuation_ratio","grammar_errors",
        "spelling_errors","stopword_ratio","capitalization_ratio",
        "flesch_reading_ease","gunning_fog_index","dale_chall_score","smog_index",
        "passive_voice_ratio","predictability_score","burstiness","sentiment_score"
    ]
    return [c for c in candidates if c in df.columns]

def tune_threshold_by_f1(y_true, proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (proba >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1

def tune_threshold_for_target_precision(y_true, proba, target_precision=0.9):
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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")

    pipe = joblib.load(MODEL_PATH)

    cfg = Config()
    df = load_raw_dataframe(cfg)
    df = basic_clean(df, cfg)

    # --- IMPORTANT: build X as a DataFrame with text + numeric columns ---
    num_cols = select_numeric_columns(df)
    feature_cols = [cfg.text_col] + num_cols
    X = df[feature_cols].copy()
    y = df[cfg.label_col].astype(int)

    # same split logic as training (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # --- Evaluate ---
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

    # threshold tuning
    best_t, best_f1 = tune_threshold_by_f1(y_test, proba)
    pred_f1 = (proba >= best_t).astype(int)
    prec_f1, rec_f1, f1_f1, _ = precision_recall_fscore_support(y_test, pred_f1, average='binary', zero_division=0)

    t_prec, prec_hp, rec_hp, f1_hp = tune_threshold_for_target_precision(y_test, proba, target_precision=0.90)

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

    # plots & reports
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - Hybrid")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    cm05 = confusion_matrix(y_test, pred05)
    cm_f1 = confusion_matrix(y_test, pred_f1)
    pd.DataFrame(cm05, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(REPORT_DIR / "confusion_0.5.csv")
    pd.DataFrame(cm_f1, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(REPORT_DIR / "confusion_bestF1.csv")

    rep05 = pd.DataFrame(classification_report(y_test, pred05, output_dict=True, zero_division=0)).T
    rep_f1 = pd.DataFrame(classification_report(y_test, pred_f1, output_dict=True, zero_division=0)).T
    rep05.to_csv(REPORT_DIR / "classification_report_0.5.csv")
    rep_f1.to_csv(REPORT_DIR / "classification_report_bestF1.csv")

    out = {
        "base_metrics": base_metrics,
        "tuned": tuned,
        "artifacts": {
            "roc_curve_png": str(REPORT_DIR / "roc_curve.png"),
            "cm_0.5_csv": str(REPORT_DIR / "confusion_0.5.csv"),
            "cm_bestF1_csv": str(REPORT_DIR / "confusion_bestF1.csv"),
            "report_0.5_csv": str(REPORT_DIR / "classification_report_0.5.csv"),
            "report_bestF1_csv": str(REPORT_DIR / "classification_report_bestF1.csv"),
        }
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(out, indent=2))

    print("\n=== Hybrid Evaluation Summary ===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
