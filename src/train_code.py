from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)

from featurizers import code_features

# -------- Config --------
DATA_PATH = Path("data/ai_human_content_detection_dataset.csv")   # point to your code dataset if separate
TEXT_COL = "text_content"
LABEL_COL = "label"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "code_tfidfchar_logreg.joblib"
METRICS_PATH = OUTPUT_DIR / "code_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "code_confusion_matrix.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "code_classification_report.csv"

# -------- Code feature transformer (shared/importable) --------
code_feat_transformer = Pipeline([
    ("extract", FunctionTransformer(code_features, validate=False)),
    ("scale", StandardScaler()),
])

# -------- Vectorizers (char + word n-grams) --------
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=80_000)
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60_000)

# Combine: [char-ngrams | word-ngrams | code-features]
pre = ColumnTransformer(
    transformers=[
        ("char", char_vec, TEXT_COL),
        ("word", word_vec, TEXT_COL),
        ("codef", code_feat_transformer, TEXT_COL),
    ],
    remainder="drop"
)

clf = LogisticRegression(
    max_iter=3000,
    solver="saga",
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

def main():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    y = df[LABEL_COL].astype(int)
    X = df[[TEXT_COL]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    report_df = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T
    cm = confusion_matrix(y_test, pred)

    joblib.dump(pipe, MODEL_PATH)
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,
        "threshold": 0.5,
        "model_path": str(MODEL_PATH),
        "used_code_features": ["comment_ratio","symbol_ratio","avg_line_len","indent_depth","camel_snake_ratio","digit_ratio"]
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, columns=["pred_0","pred_1"], index=["true_0","true_1"]).to_csv(CONFUSION_PATH, index=True)
    report_df.to_csv(CLASS_REPORT_PATH, index=True)

    print("\n=== Code Model Results (0.5) ===")
    print(json.dumps(metrics, indent=2))
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
