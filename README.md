
# AIorNOT – AI vs Human Text Classifier

A practical, end‑to‑end project for classifying whether a piece of text is **AI‑generated** or **human‑written**. This README turns the bare repository skeleton into a complete, ready‑to‑extend application with clear setup, usage, and extension paths.

> **Scope:** This guide is written to work with the current repository layout (`models/`, `src/`, `ui/`, `requirements.txt`, `sample.txt`). Where the original repo doesn’t yet include a script or module mentioned below, copy the provided examples into your repo to enable the same behavior.

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [1) Create & Activate a Virtual Environment](#1-create--activate-a-virtual-environment)
  - [2) Install Dependencies](#2-install-dependencies)
  - [3) Run a Simple Classifier (example code)](#3-run-a-simple-classifier-example-code)
  - [4) Try it on `sample.txt`](#4-try-it-on-sampletxt)
- [Command‑Line Interface (CLI)](#command-line-interface-cli)
- [Python API](#python-api)
- [Local UI (Optional)](#local-ui-optional)
- [Models](#models)
- [Datasets & Preprocessing](#datasets--preprocessing)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Tasks & Makefile (Optional)](#project-tasks--makefile-optional)
- [Testing](#testing)
- [Logging](#logging)
- [Packaging & Distribution](#packaging--distribution)
- [Deployment Ideas](#deployment-ideas)
- [Contributing](#contributing)
- [License](#license)
- [Roadmap](#roadmap)
- [FAQ](#faq)

---

## Features
- Detects whether input **text** is more likely **AI‑generated** or **human**.
- Clean separation of concerns: experiment code in `src/`, saved artifacts in `models/`, optional UI in `ui/`.
- Works offline with classical ML or transformer embeddings; can be extended to use hosted detectors.
- Example CLI and Python API included.
- Extensible evaluation and reporting helpers.

---

## Repository Structure
```
AIorNOT/
├─ models/                # Saved weights/checkpoints, vectorizers, label encoders
├─ src/                   # Core Python package (feature extraction, model, utils)
│  ├─ __init__.py
│  ├─ features.py         # Text cleaning, tokenization, TF‑IDF, embedding hooks
│  ├─ model.py            # Model definition & load/save helpers
│  ├─ predict.py          # Inference utilities
│  ├─ train.py            # (Optional) training pipeline
│  └─ utils.py            # Shared helpers (config, paths, I/O)
├─ ui/                    # Optional local UI (Streamlit/Gradio)
├─ requirements.txt       # Python dependencies
├─ sample.txt             # Example input for a quick demo
└─ README.md              # (This file)
```
> If some files don’t exist yet, copy the example snippets below into the indicated paths.

---

## Requirements
- **Python**: 3.9 or newer recommended  
- **OS**: Linux, macOS, or Windows  
- **Dependencies**: managed via `requirements.txt`

Suggested baseline dependencies (if not already present):
```txt
click>=8.1
joblib>=1.4
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
scipy>=1.13
regex>=2024.5
rich>=13.7
textstat>=0.7
unidecode>=1.3
# Optional
transformers>=4.43
sentencepiece>=0.2
torch>=2.2 ; platform_system!='Windows' or extra_index_url
beautifulsoup4>=4.12
streamlit>=1.36  # if you use the optional UI
```

---

## Quick Start

### 1) Create & Activate a Virtual Environment
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2) Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run a Simple Classifier (example code)
Create `src/features.py` with basic text features:
```python
# src/features.py
from __future__ import annotations
import re
from typing import Iterable

BASIC_PUNCT = re.compile(r"[^\w\s]")
MULTISPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    t = text.strip()
    t = t.replace("\u200b", " ")  # zero‑width
    t = BASIC_PUNCT.sub(" ", t)
    t = MULTISPACE.sub(" ", t)
    return t.lower()

def default_stopwords() -> set[str]:
    return {
        # minimal illustrative list—replace with a better list
        "the","a","an","to","and","of","in","that","is","it","for","on","as","with","this","by",
    }

def word_stats(texts: Iterable[str]) -> list[dict]:
    out = []
    for t in texts:
        n_chars = len(t)
        n_words = len(t.split())
        avg_word_len = (sum(len(w) for w in t.split()) / max(n_words, 1)) if n_words else 0
        out.append({
            "n_chars": n_chars,
            "n_words": n_words,
            "avg_word_len": avg_word_len,
        })
    return out
```

Create `src/model.py` with a baseline TF‑IDF + Logistic Regression:
```python
# src/model.py
from __future__ import annotations
import joblib
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["human", "ai"]

@dataclass
class TextClassifier:
    pipeline: Pipeline

    @classmethod
    def new(cls):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
            ("clf", LogisticRegression(max_iter=200))
        ])
        return cls(pipe)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        proba = self.pipeline.predict_proba(X)
        return proba

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str):
        pipe = joblib.load(path)
        return cls(pipe)
```

Create `src/predict.py` to run inference using a saved model or a quick zero‑shot heuristic fallback:
```python
# src/predict.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable

from .features import clean_text
from .model import TextClassifier, LABELS

DEFAULT_MODEL_PATH = Path("models/tfidf_lr.joblib")

def _fallback_score(texts: Iterable[str]):
    """Very naive heuristic if no model is trained yet.
    Returns probability of AI‑generated based on simple patterns.
    Replace when you train a model.
    """
    out = []
    for t in texts:
        t = clean_text(t)
        # toy heuristic: longer texts with low variance in word length skew AI
        words = t.split()
        ai_score = 0.5
        if len(words) > 250:
            ai_score += 0.1
        if sum(len(w) for w in words)/max(len(words),1) < 4.2:
            ai_score += 0.1
        out.append(min(max(ai_score, 0.0), 1.0))
    return out

def predict_proba_ai(texts: Iterable[str]):
    texts = list(texts)
    if DEFAULT_MODEL_PATH.exists():
        clf = TextClassifier.load(str(DEFAULT_MODEL_PATH))
        # proba order corresponds to LABELS = ["human", "ai"]
        import numpy as np
        proba = clf.predict_proba(texts)
        return proba[:, 1].tolist()
    else:
        return _fallback_score(texts)
```

Create `src/utils.py` with simple I/O helpers:
```python
# src/utils.py
from __future__ import annotations
from pathlib import Path

def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def ensure_dirs(*paths: str | Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
```

### 4) Try it on `sample.txt`
Create `scripts/predict_file.py` (new) to wire everything together:
```python
# scripts/predict_file.py
from src.predict import predict_proba_ai
from src.utils import read_text_file
import argparse, json

parser = argparse.ArgumentParser(description="AI vs Human text prediction")
parser.add_argument("path", help="Path to a .txt file")
args = parser.parse_args()

text = read_text_file(args.path)
proba_ai = predict_proba_ai([text])[0]
label = "AI" if proba_ai >= 0.5 else "Human"
print(json.dumps({
    "file": args.path,
    "label": label,
    "prob_ai": round(proba_ai, 4)
}, ensure_ascii=False))
```
Run it:
```bash
python scripts/predict_file.py sample.txt
```
You’ll get a JSON prediction like:
```json
{"file":"sample.txt","label":"Human","prob_ai":0.2671}
```

> **Tip:** Once you’ve trained a model and saved it to `models/tfidf_lr.joblib`, the prediction will use that model automatically; otherwise it falls back to the quick heuristic.

---

## Command‑Line Interface (CLI)
Add `cli.py` to `src/` for a nicer multi‑command interface:
```python
# src/cli.py
import json
import click
from pathlib import Path
from .predict import predict_proba_ai

@click.group()
def cli():
    """AIorNOT command‑line tools."""

@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def predict(path: str):
    """Predict AI vs Human for a text file."""
    text = Path(path).read_text(encoding="utf-8")
    p = predict_proba_ai([text])[0]
    label = "AI" if p >= 0.5 else "Human"
    click.echo(json.dumps({"file": path, "label": label, "prob_ai": round(p, 4)}))

if __name__ == "__main__":
    cli()
```
Usage:
```bash
python -m src.cli predict sample.txt
```

Optional: make an executable entry‑point by adding this to `pyproject.toml` (if you use one):
```toml
[project.scripts]
aiorn = "src.cli:cli"
```
Then:
```bash
aiorn predict sample.txt
```

---

## Python API
```python
from src.predict import predict_proba_ai

text = "Your test paragraph goes here ..."
prob_ai = predict_proba_ai([text])[0]
print(prob_ai)
```

---

## Local UI (Optional)
Create a simple **Streamlit** app at `ui/app.py`:
```python
# ui/app.py
import streamlit as st
from src.predict import predict_proba_ai

st.set_page_config(page_title="AIorNOT – Text Classifier", page_icon="🤖", layout="centered")
st.title("AIorNOT – Text Classifier")
text = st.text_area("Paste text to analyze", height=220)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        p = predict_proba_ai([text])[0]
        label = "AI" if p >= 0.5 else "Human"
        st.metric("Prediction", label)
        st.write(f"**Probability (AI):** {p:.4f}")
```
Run:
```bash
streamlit run ui/app.py
```

---

## Models
Two suggested paths:
1. **Classical ML baseline** (ships with this doc): TF‑IDF + Logistic Regression. Good for quick local testing.
2. **Embeddings + simple classifier**: Generate sentence embeddings (e.g., using `transformers`) and train a small classifier.

Save trained artifacts to `models/` using `joblib` so inference is reproducible.

---

## Datasets & Preprocessing
- Collect balanced corpora of human‑written and AI‑generated texts.
- Clean (normalize whitespace, strip control chars), tokenize, optionally remove stopwords.
- Ensure minimum length (e.g., 250 chars) to reduce noise.
- Split train/valid/test with author/source disjointness to avoid leakage.

Add a minimal training loop in `src/train.py` (sketch):
```python
# src/train.py (sketch)
from .model import TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

texts, labels = load_your_dataset()  # TODO: implement
X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42)
clf = TextClassifier.new().fit(X_tr, y_tr)
print(classification_report(y_te, clf.pipeline.predict(X_te)))
clf.save("models/tfidf_lr.joblib")
```

---

## Evaluation
Report at least:
- Accuracy, Precision/Recall/F1 (macro)
- ROC‑AUC
- Calibration (optional)
- Robustness checks: paraphrasing, minor noise, punctuation removal

Provide a `notebooks/` folder for exploratory analysis if desired.

---

## Configuration
Recommended `pyproject.toml` or `setup.cfg` for tooling (ruff/flake8, black, isort, mypy). Example `ruff` config:
```toml
[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP"]
```

---

## Project Tasks & Makefile (Optional)
Create a `Makefile` to standardize workflows:
```makefile
.PHONY: setup fmt lint test ui
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
fmt:
	ruff --fix . || true
	black . || true
lint:
	ruff .
	black --check .
test:
	pytest -q
ui:
	streamlit run ui/app.py
```

---

## Testing
Use `pytest` with fixtures for sample texts; add regression tests to lock model/feature behavior.

---

## Logging
Use `rich` or `logging` for structured logs; write key metrics and predictions to JSONL for auditability.

---

## Packaging & Distribution
- Convert `src/` into a proper package (e.g., `aion`)
- Include `pyproject.toml`
- Optional: publish to a private index if you want to reuse across services

---

## Deployment Ideas
- **Local API** with FastAPI:
  - Create `api/main.py` exposing `/predict` that accepts raw text and returns label + probability.
- **Docker**: build a small image with the model artifact baked in.
- **Serverless**: package `predict_proba_ai` into an AWS Lambda / Cloud Run function.

---

## Contributing
1. Fork & create a feature branch
2. Add/modify code and tests
3. Ensure `make lint test` passes
4. Open a PR with a concise description and examples

---

## License
# SPDX-License-Identifier: MIT

---

## Roadmap
- [ ] Replace heuristic fallback with a trained baseline (saved to `models/`)
- [ ] Add FastAPI endpoint & Dockerfile
- [ ] Add real evaluation reports and benchmark datasets
- [ ] Optional: add integration with hosted detectors (e.g., call an external API) with rate‑limit/backoff
- [ ] CI (GitHub Actions) for lint/test

---

## FAQ
**Does this require internet access?**  
No. The baseline runs locally. Optional hosted detectors would require network access and API keys.

**Is there a minimum text length?**  
For best signal, aim for at least ~250 characters; extremely short texts are noisy for any detector.

**Can I use transformer embeddings instead of TF‑IDF?**  
Yes—generate embeddings (e.g., `sentence-transformers`) and train a small classifier; swap it into `TextClassifier`.

**How do I reproduce predictions?**  
Pin dependencies, freeze random seeds, and save your vectorizer + model (as we do via `joblib`).

---

### Attribution & Notes
- This README includes concrete code scaffolding to make the repository runnable even if some scripts aren’t present yet. Replace the example pieces with your actual training/inference code when available.
