from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Config ----------
@dataclass(frozen=True)
class Config:
    data_path: str = os.environ.get(
        "DATA_PATH",
        "data/ai_human_content_detection_dataset.csv"
    )
    text_col: str = "text_content"
    label_col: str = "label"
    test_size: float = 0.2
    random_state: int = 42

# ---------- Loaders ----------
def load_raw_dataframe(cfg: Config = Config()) -> pd.DataFrame:
    """Load the CSV and perform minimal validation."""
    df = pd.read_csv(cfg.data_path)
    _validate_columns(df, cfg)
    return df

def _validate_columns(df: pd.DataFrame, cfg: Config) -> None:
    required = {cfg.text_col, cfg.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ---------- Cleaning ----------
def basic_clean(df: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Lightweight cleaning suitable for a baseline model."""
    df = df.copy()
    # Fill text NAs and standardize types
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)

    # label to int (assumes 0/1)
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    return df

# ---------- Splitting ----------
def get_xy(df: pd.DataFrame, cfg: Config = Config()) -> Tuple[pd.Series, pd.Series]:
    X = df[cfg.text_col]
    y = df[cfg.label_col]
    return X, y

def train_test_split_xy(
    X: pd.Series,
    y: pd.Series,
    cfg: Config = Config()
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified split keeps label ratio similar in train and test."""
    return train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

# ---------- Quick EDA helpers (optional) ----------
def label_distribution(df: pd.DataFrame, cfg: Config = Config()) -> pd.Series:
    return df[cfg.label_col].value_counts(normalize=False)

def label_distribution_pct(df: pd.DataFrame, cfg: Config = Config()) -> pd.Series:
    return (df[cfg.label_col].value_counts(normalize=True) * 100).round(2)
