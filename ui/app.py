from __future__ import annotations
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# =============================
# Config / Paths
# =============================
MODEL_OPTIONS = {
    # TEXT models
    "Text (Baseline TF-IDF + LR)": {
        "model_path": Path("models/baseline_tfidf_logreg.joblib"),
        "eval_summary_path": Path("models/eval/evaluation_summary.json"),
        "supports_token_explain": True,
        "type": "text",
    },
    "Text (Hybrid TF-IDF→SVD + numeric + LR)": {
        "model_path": Path("models/hybrid_svd_num_logreg.joblib"),
        "eval_summary_path": Path("models/eval_hybrid/evaluation_summary.json"),
        "supports_token_explain": False,
        "type": "text",
    },
    # CODE model
    "Code (Char+Word TF-IDF + LR)": {
        "model_path": Path("models/code_tfidfchar_logreg.joblib"),
        "eval_summary_path": Path("models/eval_code/evaluation_summary.json"),
        "supports_token_explain": False,  # not token-interpretable
        "type": "code",
    },
}

HYBRID_METRICS_PATH = Path("models/hybrid_metrics.json")

# Limits
MAX_CHARS = 15000          # cap extracted text to keep inference fast
MAX_FILE_SIZE_MB = 10      # per-file upload limit

# Code support
CODE_EXTS = [
    "py","js","ts","jsx","tsx","java","kt","swift","cpp","cc","c","cs","go","rs",
    "php","rb","sh","ps1","bat","r","m","scala","lua","pl","sql","html","css","xml",
    "json","yml","yaml","toml","md","ini","ipynb"
]
CODE_box = [
    "py.......ect",
]
Auto_box = [
    "pdf , txt ect ect",
]
EXT_TO_LANG = {
    "py":"python","js":"javascript","ts":"typescript","jsx":"jsx","tsx":"tsx","java":"java","kt":"kotlin","swift":"swift",
    "cpp":"cpp","cc":"cpp","c":"c","cs":"csharp","go":"go","rs":"rust","php":"php","rb":"ruby","sh":"bash","ps1":"powershell",
    "bat":"batch","r":"r","m":"objectivec","scala":"scala","lua":"lua","pl":"perl","sql":"sql","html":"html","css":"css",
    "xml":"xml","json":"json","yml":"yaml","yaml":"yaml","toml":"toml","md":"markdown","ini":"ini","ipynb":"json"
}

# Upload type groups
TEXT_EXTS = ["pdf", "docx", "txt"]
ALL_UPLOAD_EXTS = sorted(set(TEXT_EXTS + CODE_EXTS))  # everything

def is_code_file(name: str) -> bool:
    ext = name.split(".")[-1].lower() if "." in name else ""
    return ext in CODE_EXTS

def lang_from_name(name: str) -> str:
    ext = name.split(".")[-1].lower() if "." in name else ""
    return EXT_TO_LANG.get(ext, "text")

# =============================
# Cached loaders
# =============================
@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
    return joblib.load(model_path)

@st.cache_resource
def load_eval_summary(eval_summary_path: Path):
    if not eval_summary_path.exists():
        # sensible defaults if eval wasn’t run yet
        return {
            "base_metrics": {"threshold": 0.5},
            "tuned": {"best_f1_threshold": 0.5, "high_precision_threshold_0.90": 0.9},
        }
    return json.loads(eval_summary_path.read_text())

# =============================
# Threshold resolver
# =============================
def resolve_threshold(mode: str, custom: float | None, eval_summary: dict) -> float:
    if custom is not None:
        return float(custom)
    if mode == "Default (0.5)":
        return float(eval_summary.get("base_metrics", {}).get("threshold", 0.5))
    if mode == "Best-F1 (from eval)":
        return float(eval_summary.get("tuned", {}).get("best_f1_threshold", 0.5))
    if mode == "High-Precision (~0.90)":
        return float(eval_summary.get("tuned", {}).get("high_precision_threshold_0.90", 0.9))
    return 0.5

# =============================
# Smart predictors (handle 1D vs 2D/DataFrame inputs)
# =============================
def predict_proba_single(model, text: str) -> float:
    """
    Try common input formats so we work with pipelines that expect either
    a 1D list of docs or a DataFrame with specific columns.
    """
    candidates = [
        [text],  # typical TfidfVectorizer pipelines
        pd.DataFrame({"text_content": [text]}),
        pd.DataFrame({"code_content": [text]}),
        pd.DataFrame({"text": [text]}),
        pd.DataFrame({"content": [text]}),
    ]
    last_err = None
    for X in candidates:
        try:
            return float(model.predict_proba(X)[:, 1][0])
        except Exception as e:
            last_err = e
            continue
    # If everything failed, bubble up the last error
    raise last_err if last_err else RuntimeError("predict_proba failed for all input shapes")

def predict_proba_batch(model, texts: list[str]) -> np.ndarray:
    """
    Batch version for CSV tab. Returns a 1D np.ndarray of probs.
    """
    candidates = [
        texts,
        pd.DataFrame({"text_content": texts}),
        pd.DataFrame({"code_content": texts}),
        pd.DataFrame({"text": texts}),
        pd.DataFrame({"content": texts}),
    ]
    last_err = None
    for X in candidates:
        try:
            return model.predict_proba(X)[:, 1]
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("predict_proba (batch) failed for all input shapes")

# =============================
# Explanation helpers
# =============================
def explain_baseline_example(model, text: str, top_k: int = 15):
    """
    Per-example token contributions for TF-IDF + Logistic Regression:
    contribution = coef[token] * tfidf_value[token]
    Returns (df_pos, df_neg).
    """
    vec = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    if vec is None or clf is None:
        # Not a plain tfidf+logreg pipeline; skip token explanation
        return pd.DataFrame({"token": [], "contribution": []}), pd.DataFrame({"token": [], "contribution": []})

    X = vec.transform([text])  # sparse row
    feats = vec.get_feature_names_out()
    coefs = clf.coef_.ravel()

    X_csr = X.tocoo()
    contribs = {}
    for i, v in zip(X_csr.col, X_csr.data):
        contribs[i] = coefs[i] * v

    if not contribs:
        empty = pd.DataFrame({"token": [], "contribution": []})
        return empty, empty

    idxs = np.fromiter(contribs.keys(), dtype=int)
    vals = np.fromiter(contribs.values(), dtype=float)

    order_pos = np.argsort(vals)[-top_k:][::-1]
    order_neg = np.argsort(vals)[:top_k]

    df_pos = pd.DataFrame({"token": feats[idxs[order_pos]], "contribution": vals[order_pos]})
    df_neg = pd.DataFrame({"token": feats[idxs[order_neg]], "contribution": vals[order_neg]})
    return df_pos, df_neg

def load_hybrid_numcols_and_svd(model, metrics_path: Path):
    """
    Read numeric feature names from training metrics JSON and the SVD dimension
    from the pipeline.
    """
    used_numeric: list[str] = []
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        used_numeric = m.get("used_numeric_features", [])

    pre = model.named_steps["pre"]
    svd = pre.named_transformers_["text"].named_steps["svd"]
    n_svd = int(svd.n_components)
    return used_numeric, n_svd

def explain_hybrid_example(model, text: str, used_numeric: list[str], n_svd: int):
    """
    Per-example contributions in the hybrid transformed space:
    - Sum all SVD components as one "Text (SVD)" contribution (not token-interpretable)
    - Show each numeric feature's standardized value and its contribution
    """
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    coef = clf.coef_.ravel()

    # Build a single-row DataFrame with required columns
    row = {"text_content": text}
    for c in used_numeric:
        row[c] = np.nan  # imputed inside pipeline
    X_one = pd.DataFrame([row])

    z = pre.transform(X_one)
    # Ensure dense
    try:
        z = np.asarray(z).ravel()
    except Exception:
        z = z.toarray().ravel()

    contrib = coef * z

    svd_contrib_sum = float(contrib[:n_svd].sum())
    num_contrib = contrib[n_svd:]
    num_values = z[n_svd:]

    df_num = pd.DataFrame({
        "feature": used_numeric,
        "std_value": num_values,
        "contribution": num_contrib
    }).sort_values("contribution", ascending=False).reset_index(drop=True)

    return svd_contrib_sum, df_num


# =============================
# Parsing helpers (PDF/DOCX/TXT/Code/Notebooks)
# =============================
def human_size(num_bytes: int) -> str:
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.0f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def file_badge(name: str) -> str:
    ext = name.split(".")[-1].lower() if "." in name else ""
    if is_code_file(name):
        emoji = "💻"
    else:
        emoji = {"pdf": "📄", "docx": "📝", "txt": "📜"}.get(ext, "📦")
    return f"{emoji} `{name}`"

def excerpt(text: str, n: int = 320) -> str:
    t = (text or "").strip().replace("\r", " ")
    return t if len(t) <= n else (t[:n] + " …")

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def read_docx(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    return "\n".join([p.text for p in doc.paragraphs])

def read_code(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def read_ipynb(file_bytes: bytes) -> str:
    # Extract code cells from a notebook
    try:
        nb = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        cells = nb.get("cells", [])
        lines: list[str] = []
        for c in cells:
            if c.get("cell_type") == "code":
                src = c.get("source", [])
                if isinstance(src, list):
                    lines.extend(src)
                elif isinstance(src, str):
                    lines.append(src)
                lines.append("\n")
        return "".join(lines)
    except Exception:
        return read_code(file_bytes)

def extract_text_from_upload(upload) -> str:
    name = (upload.name or "").lower()
    data = upload.read()
    if name.endswith(".pdf"):
        text = read_pdf(data)
    elif name.endswith(".docx"):
        text = read_docx(data)
    elif name.endswith(".ipynb"):
        text = read_ipynb(data)
    elif name.endswith(".txt"):
        text = read_txt(data)
    elif is_code_file(name):
        text = read_code(data)
    else:
        text = read_txt(data)  # fallback
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text.strip()

def pick_model_key_for_content(content_type: str, current_key: str) -> str:
    if content_type == "Text":
        if MODEL_OPTIONS[current_key]["type"] == "text":
            return current_key
        return "Text (Baseline TF-IDF + LR)"
    if content_type == "Code":
        return "Code (Char+Word TF-IDF + LR)"
    return current_key  # Auto (paste/CSV assume text)

def _humanize_ext_list(exts: list[str], limit: int = 14) -> str:
    xs = sorted({e.lower() for e in exts})
    if len(xs) <= limit:
        return ", ".join(xs)
    return f"{', '.join(xs[:limit])}, +{len(xs)-limit} more"
# =============================
# UI
# =============================
st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖", layout="centered")
st.title("🤖 AI vs Human Text Detector")

with st.sidebar:
    st.header("Model & Thresholds")
    content_choice = st.radio("Content type", ["Auto (assume Text)", "Text", "Code"], index=0)
    content_type = "Auto" if content_choice.startswith("Auto") else content_choice

    # Let the user pick a model, but smartly switch if they chose "Code"
    model_key = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()), index=0)
    effective_model_key = pick_model_key_for_content(
        "Text" if content_type == "Auto" else content_type,
        model_key,
    )
    selected = MODEL_OPTIONS[effective_model_key]

    st.caption(f"**Model:** `{selected['model_path']}`")
    st.caption(f"**Eval summary:** `{selected['eval_summary_path']}`")

    mode = st.selectbox(
        "Threshold mode",
        ["Default (0.5)", "Best-F1 (from eval)", "High-Precision (~0.90)"],
        index=0,
    )
    use_custom = st.checkbox("Use custom threshold", value=False)
    custom_thr = st.slider("Custom threshold", 0.0, 1.0, 0.5, 0.01) if use_custom else None

# Load artifacts
model = load_model(selected["model_path"])
eval_summary = load_eval_summary(selected["eval_summary_path"])
thr = resolve_threshold(mode, custom_thr, eval_summary)
st.write(f"**Decision threshold:** {thr:.3f}")

# Tabs
tab_paste, tab_upload, tab_csv = st.tabs(["Paste text", "Upload files", "Batch CSV"])

# -----------------------------
# Paste Tab
# -----------------------------
with tab_paste:
    text = st.text_area("Paste text here:", height=220, placeholder="Type or paste any paragraph...")
    if st.button("Classify (pasted text)", use_container_width=True):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            proba = predict_proba_single(model, text)
            label = "AI" if proba >= thr else "Human"

            st.subheader("Result")
            st.metric("Prediction", label)
            st.progress(min(1.0, proba))
            st.write(f"**Probability (AI):** {proba:.4f}")
            st.caption(f"Model: `{selected['model_path']}` | Mode: {mode} | Threshold: {thr:.3f}")

            st.divider()
            with st.expander("🔎 Explain this prediction"):
                if selected["supports_token_explain"]:
                    try:
                        df_pos, df_neg = explain_baseline_example(model, text, top_k=15)
                        if len(df_pos) == 0 and len(df_neg) == 0:
                            st.info("No token contributions to show (very short or stopword-only text).")
                        else:
                            st.write("**Top tokens pushing AI**")
                            st.dataframe(df_pos, use_container_width=True, hide_index=True)
                            st.write("**Top tokens pushing Human**")
                            st.dataframe(df_neg, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.info(f"Token explanation unavailable: {e}")
                else:
                    try:
                        used_numeric, n_svd = load_hybrid_numcols_and_svd(model, HYBRID_METRICS_PATH)
                        svd_sum, df_num = explain_hybrid_example(model, text, used_numeric, n_svd)
                        st.write(f"**Text (SVD) aggregate contribution:** {svd_sum:.6f}")
                        if not df_num.empty:
                            st.write("**Numeric feature contributions (standardized value & contribution)**")
                            st.dataframe(df_num, use_container_width=True, hide_index=True)
                        else:
                            st.info("No numeric features found in hybrid metrics.")
                    except Exception as e:
                        st.info(f"Hybrid explanation unavailable: {e}")

# -----------------------------
# Upload Tab (mode-aware types, tidy UI)
# -----------------------------
with tab_upload:
    st.subheader("Upload files")

    # Decide which extensions to allow based on sidebar content_type
    if content_type == "Text":
        allowed_exts = TEXT_EXTS
        accept_label = "PDF, DOCX, TXT"
    elif content_type == "Code":
        allowed_exts = CODE_box
        accept_label = "Code & Notebooks"
    else:
        # Auto: allow all
        allowed_exts = Auto_box
        accept_label = "Text, Code & Notebooks"

    # Header row with a small hint + a button that reveals the full list
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.caption(f"Accepting: **{accept_label}**")
    with c_right:
        # Compact disclosure of full supported types
        try:
            with st.popover("Supported types"):
                st.markdown("**Text:** `pdf`, `docx`, `txt`")
                st.markdown("**Code & Notebooks:**")
                st.code(_humanize_ext_list(CODE_EXTS), language="markdown")
        except Exception:
            with st.expander("Supported types"):
                st.markdown("**Text:** `pdf`, `docx`, `txt`")
                st.markdown("**Code & Notebooks:**")
                st.code(_humanize_ext_list(CODE_EXTS), language="markdown")

    uploads = st.file_uploader(
        "Drag & drop or browse (multiple allowed)",
        type=allowed_exts,
        accept_multiple_files=True,
        help="",
    )

    if st.button("Classify uploaded file(s)", use_container_width=True):
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            rows: list[dict] = []
            progress = st.progress(0.0)
            total = len(uploads)

            for idx, f in enumerate(uploads, start=1):
                name = f.name
                size = len(f.getbuffer())
                ext = (name.split(".")[-1].lower() if "." in name else "")
                size_ok = size <= MAX_FILE_SIZE_MB * 1024 * 1024

                # Is this file considered "code"?
                # (In Text mode, you won't get code files because the UI disallows them.)
                is_code = is_code_file(name) or name.lower().endswith(".ipynb")

                with st.container(border=True):
                    st.markdown(f"<div class='file-card'>**{file_badge(name)}** · {human_size(size)}</div>",
                                unsafe_allow_html=True)

                    if not size_ok:
                        st.error(f"File too large (> {MAX_FILE_SIZE_MB} MB). Skipped.")
                        rows.append({
                            "file": name, "type": ext, "size_bytes": size,
                            "chars": np.nan, "prob_ai": np.nan,
                            "prediction": "Error: too large"
                        })
                        progress.progress(min(idx / total, 1.0))
                        continue

                    try:
                        # Parse text from the file
                        f.seek(0)
                        text = extract_text_from_upload(f)
                        text = (text or "").strip()
                        if len(text) > MAX_CHARS:
                            st.info(f"Text too long; truncated to {MAX_CHARS} characters for speed.")
                            text = text[:MAX_CHARS]

                        if not text:
                            st.warning("No extractable text found.")
                            rows.append({
                                "file": name, "type": ext, "size_bytes": size,
                                "chars": 0, "prob_ai": np.nan, "prediction": "Empty/Unparsable"
                            })
                            progress.progress(min(idx / total, 1.0))
                            continue

                        # Choose model:
                        # - In Code mode, force Code model
                        # - In Text mode, force Text model
                        # - In Auto, choose per file by extension
                        if content_type == "Code":
                            per_content = "Code"
                        elif content_type == "Text":
                            per_content = "Text"
                        else:
                            per_content = "Code" if is_code else "Text"

                        per_key = pick_model_key_for_content(per_content, effective_model_key)
                        per_selected = MODEL_OPTIONS[per_key]

                        per_model = load_model(per_selected["model_path"])
                        per_eval = load_eval_summary(per_selected["eval_summary_path"])
                        per_thr = resolve_threshold(mode, custom_thr, per_eval)

                        p = predict_proba_single(per_model, text)
                        pred = "AI" if p >= per_thr else "Human"

                        cA, cB, cC, cD = st.columns([2, 1, 1, 1])
                        with cA: st.write(f"**Prediction:** {pred}")
                        with cB: st.write(f"**AI prob:** {p:.4f}")
                        with cC: st.write(f"**Chars:** {len(text)}")
                        with cD: st.write(f"**Threshold:** {per_thr:.3f}")

                        # Clean preview: slider height, optional wrap, download extracted text
                        with st.expander("Preview / Export extracted text"):
                            # controls
                            c1, c2, c3 = st.columns([2,1,1])
                            with c1:
                                lines_to_show = st.slider("Preview lines", 30, 400, 120, 10, key=f"lines-{idx}")
                            with c2:
                                show_full = st.toggle("Full text", value=False, key=f"full-{idx}")
                            with c3:
                                wrap = st.toggle("Wrap", value=False, key=f"wrap-{idx}")

                            display = text if show_full else "\n".join(text.splitlines()[:lines_to_show])
                            st.text_area(
                                "Preview",
                                value=display,
                                height=min(900, max(220, 12 * (display.count('\n') + 4))),
                                key=f"ta-{idx}",
                                label_visibility="collapsed",
                            )
                            st.download_button(
                                "Download extracted text",
                                data=text.encode("utf-8"),
                                file_name=(Path(name).stem + "_extracted.txt"),
                                mime="text/plain",
                                use_container_width=True,
                                key=f"dl-{idx}",
                            )

                        # optional explanations
                        with st.expander("Explain this file"):
                            try:
                                if per_selected["supports_token_explain"]:
                                    df_pos, df_neg = explain_baseline_example(per_model, text, top_k=12)
                                    if len(df_pos) == 0 and len(df_neg) == 0:
                                        st.info("No token contributions to show.")
                                    else:
                                        st.write("**Top tokens pushing AI**")
                                        st.dataframe(df_pos, use_container_width=True, hide_index=True)
                                        st.write("**Top tokens pushing Human**")
                                        st.dataframe(df_neg, use_container_width=True, hide_index=True)
                                else:
                                    used_numeric, n_svd = load_hybrid_numcols_and_svd(per_model, HYBRID_METRICS_PATH)
                                    svd_sum, df_num = explain_hybrid_example(per_model, text, used_numeric, n_svd)
                                    st.write(f"**Text (SVD) aggregate contribution:** {svd_sum:.6f}")
                                    if not df_num.empty:
                                        st.write("**Numeric feature contributions**")
                                        st.dataframe(df_num, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No numeric features available.")
                            except Exception as e:
                                st.info(f"Explanation unavailable: {e}")

                        rows.append({
                            "file": name,
                            "type": ext,
                            "size_bytes": size,
                            "chars": len(text),
                            "prob_ai": p,
                            "prediction": pred,
                            "model_used": per_key,
                        })

                    except Exception as e:
                        st.error(f"Failed to process: {e}")
                        rows.append({
                            "file": name, "type": ext, "size_bytes": size,
                            "chars": np.nan, "prob_ai": np.nan, "prediction": f"Error: {e}"
                        })

                progress.progress(min(idx / total, 1.0))

            if rows:
                df_out = pd.DataFrame(rows)
                st.subheader("File results")
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results CSV",
                    df_out.to_csv(index=False),
                    file_name="predictions_files.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# -----------------------------
# Batch CSV Tab
# -----------------------------
with tab_csv:
    st.write("Upload a CSV with a column named **text** for batch scoring.")
    csv_file = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
    if st.button("Classify CSV", use_container_width=True):
        if not csv_file:
            st.warning("Please upload a CSV.")
        else:
            try:
                df_csv = pd.read_csv(csv_file)
                if "text" not in df_csv.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    texts = df_csv["text"].fillna("").astype(str).tolist()
                    texts = [t[:MAX_CHARS] for t in texts]   # cap long inputs
                    probs = predict_proba_batch(model, texts)
                    preds = np.where(probs >= thr, "AI", "Human")
                    out = df_csv.copy()
                    out["prob_ai"] = probs
                    out["prediction"] = preds
                    st.subheader("Batch results")
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "Download results CSV",
                        out.to_csv(index=False),
                        file_name="predictions_batch.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")
