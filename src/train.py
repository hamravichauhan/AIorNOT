from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Local utils
from utils import Config, load_raw_dataframe, basic_clean, get_xy, train_test_split_xy

# -----------------------------
# Training configuration
# -----------------------------
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "baseline_tfidf_logreg.joblib"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "baseline_confusion_matrix.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "baseline_classification_report.csv"

# -----------------------------
# Build the baseline pipeline
# -----------------------------
def build_pipeline() -> Pipeline:
    """
    Why these hyperparams?
    - ngram_range=(1,2): unigrams + bigrams capture short phrases/style.
    - min_df=2: drop ultra-rare tokens (noise).
    - max_features=50000: bounds memory/overfit risk on small–medium data.
    - class_weight='balanced': handles class imbalance if present.
    - solver='saga': efficient with large sparse matrices.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            lowercase=True,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            n_jobs=1,
            random_state=42
        ))
    ])

def main():
    cfg = Config()

    # 1) Load & clean
    df = load_raw_dataframe(cfg)
    df = basic_clean(df, cfg)
    X, y = get_xy(df, cfg)

    # 2) Split (stratified keeps label ratio)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, cfg)

    # 3) Train
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 4) Evaluate
    proba_test = pipe.predict_proba(X_test)[:, 1]
    # Threshold = 0.5 for baseline; we can tune later by business needs
    y_pred = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_test, proba_test)
    except Exception:
        auc = float("nan")

    report_df = pd.DataFrame(classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )).T
    cm = confusion_matrix(y_test, y_pred)

    # 5) Save artifacts
    joblib.dump(pipe, MODEL_PATH)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,  # NaN-safe
        "threshold": 0.5,
        "model_path": str(MODEL_PATH)
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(CONFUSION_PATH, index=True)
    report_df.to_csv(CLASS_REPORT_PATH, index=True)

    # 6) Console summary
    print("\n=== Baseline Results ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report saved to:", CLASS_REPORT_PATH)
    print("Confusion matrix saved to:", CONFUSION_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
