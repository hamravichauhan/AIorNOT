# src/train_hybrid.py (fixed)

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

from utils import Config, load_raw_dataframe, basic_clean  # note: we won't use get_xy here

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "hybrid_svd_num_logreg.joblib"
METRICS_PATH = OUTPUT_DIR / "hybrid_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "hybrid_confusion_matrix.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "hybrid_classification_report.csv"

def select_numeric_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "word_count","character_count","sentence_count","lexical_diversity",
        "avg_sentence_length","avg_word_length","punctuation_ratio","grammar_errors",
        "spelling_errors","stopword_ratio","capitalization_ratio",
        "flesch_reading_ease","gunning_fog_index","dale_chall_score","smog_index",
        "passive_voice_ratio","predictability_score","burstiness","sentiment_score"
    ]
    return [c for c in candidates if c in df.columns]

def build_pipeline(text_col: str, num_cols: list[str]) -> Pipeline:
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=60_000,
            lowercase=True,
            strip_accents="unicode"
        )),
        ("svd", TruncatedSVD(n_components=300, random_state=42))
    ])

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer([
        ("text", text_pipe, text_col),
        ("num",  num_pipe,  num_cols)
    ])

    clf = LogisticRegression(
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    )

    return Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

def main():
    cfg = Config()
    df = load_raw_dataframe(cfg)
    df = basic_clean(df, cfg)

    # --- KEY FIX: build X as a DataFrame with BOTH text and numeric columns ---
    num_cols = select_numeric_columns(df)
    feature_cols = [cfg.text_col] + num_cols  # ensure both branches exist
    X = df[feature_cols].copy()               # 2D DataFrame for ColumnTransformer
    y = df[cfg.label_col].astype(int)

    # Stratified split on y; X remains a DataFrame with named columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipe = build_pipeline(cfg.text_col, num_cols)
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

    # Save artifacts
    joblib.dump(pipe, MODEL_PATH)
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,
        "threshold": 0.5,
        "model_path": str(MODEL_PATH),
        "used_numeric_features": num_cols
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, columns=["pred_0","pred_1"], index=["true_0","true_1"]).to_csv(CONFUSION_PATH, index=True)
    report_df.to_csv(CLASS_REPORT_PATH, index=True)

    print("\n=== Hybrid Results (0.5 threshold) ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report saved to:", CLASS_REPORT_PATH)
    print("Confusion matrix saved to:", CONFUSION_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
