from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import joblib

# ---------------------------
# Defaults for each type
# ---------------------------
DEFAULTS = {
    "text": {
        "model": Path("models/baseline_tfidf_logreg.joblib"),
        "eval":  Path("models/eval/evaluation_summary.json"),
    },
    "code": {
        "model": Path("models/code_tfidfchar_logreg.joblib"),
        "eval":  Path("models/eval_code/evaluation_summary.json"),
    },
}

# File extensions that we treat as "code" for auto mode
CODE_EXTS = {
    "py","js","ts","jsx","tsx","java","kt","swift","cpp","cc","c","cs","go","rs",
    "php","rb","sh","ps1","bat","r","m","scala","lua","pl","sql","html","css","xml",
    "json","yml","yaml","toml","md","ini","ipynb"
}

def infer_type_from_path(p: Path) -> str:
    ext = p.suffix.lower().lstrip(".")
    return "code" if ext in CODE_EXTS else "text"

# ---------------------------
# I/O + model loading
# ---------------------------
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first or pass --model-path.")
    return joblib.load(model_path)

def load_threshold(mode: str | None, explicit_threshold: float | None, eval_summary_path: Path) -> float:
    """
    Priority:
      1) explicit_threshold if provided
      2) mode in {"default", "best_f1", "high_precision"} read from eval_summary_path
      3) fallback to 0.5
    """
    # 1) Explicit wins
    if explicit_threshold is not None:
        return float(explicit_threshold)

    # 2) From eval summary (if available)
    if mode:
        if not eval_summary_path.exists():
            return 0.5
        data = json.loads(eval_summary_path.read_text())
        if mode == "default":
            return float(data.get("base_metrics", {}).get("threshold", 0.5))
        elif mode == "best_f1":
            return float(data.get("tuned", {}).get("best_f1_threshold", 0.5))
        elif mode == "high_precision":
            return float(data.get("tuned", {}).get("high_precision_threshold_0.90", 0.9))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # 3) Fallback
    return 0.5

def read_text_from_args_or_stdin(args):
    if args.text is not None:
        return args.text
    if args.file is not None:
        p = Path(args.file)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore")
    # If neither provided, read from stdin (pipe)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --file, or pipe text via STDIN.")

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="AI vs Human classifier (text/code)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--text", type=str, help="Raw text/code to classify")
    g.add_argument("--file", type=str, help="Path to a text/code file to classify")

    # Content-type routing
    parser.add_argument("--content-type", type=str, choices=["auto", "text", "code"], default="auto",
                        help="Which model to use. 'auto' infers from file extension; for --text/STDIN falls back to 'text'.")

    # Paths (kept for backward-compat / custom models)
    parser.add_argument("--model-path", type=str,
                        default=str(DEFAULTS["text"]["model"]),
                        help="Path to a saved sklearn Pipeline (.joblib). "
                             "Defaults to text model; auto-switches to code model if --content-type=auto and file looks like code.")
    parser.add_argument("--eval-summary-path", type=str,
                        default=str(DEFAULTS["text"]["eval"]),
                        help="Path to evaluation_summary.json used to resolve threshold modes. "
                             "Defaults to text eval; auto-switches to code eval if needed.")

    # Thresholding
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (0..1). Overrides --mode if provided.")
    parser.add_argument("--mode", type=str, choices=["default", "best_f1", "high_precision"],
                        default="default",
                        help="Threshold mode: default(0.5), best_f1 (from eval), high_precision (~0.90 precision).")

    parser.add_argument("--show-prob", action="store_true", help="Print probability as well.")
    args = parser.parse_args()

    # Resolve content type
    resolved_type = args.content_type
    if resolved_type == "auto":
        if args.file:
            resolved_type = infer_type_from_path(Path(args.file))
        else:
            resolved_type = "text"  # no filename to inspect

    # If user kept default text paths, auto-swap them to the right defaults for the resolved type.
    # If they passed custom paths, we respect them.
    model_path = Path(args.model_path)
    eval_summary_path = Path(args.eval_summary_path)

    text_defaults = DEFAULTS["text"]
    code_defaults = DEFAULTS["code"]

    def is_default_text_paths(mp: Path, ep: Path) -> bool:
        return mp == text_defaults["model"] and ep == text_defaults["eval"]

    def is_default_code_paths(mp: Path, ep: Path) -> bool:
        return mp == code_defaults["model"] and ep == code_defaults["eval"]

    if resolved_type == "code" and is_default_text_paths(model_path, eval_summary_path):
        # Switch to code defaults
        model_path = code_defaults["model"]
        eval_summary_path = code_defaults["eval"]
    elif resolved_type == "text" and is_default_code_paths(model_path, eval_summary_path):
        # Switch to text defaults
        model_path = text_defaults["model"]
        eval_summary_path = text_defaults["eval"]
    # Else: custom paths provided → keep as-is.

    # Read input text
    text = read_text_from_args_or_stdin(args)

    # Load & predict
    model = load_model(model_path)
    thr = load_threshold(args.mode, args.threshold, eval_summary_path)

    proba = float(model.predict_proba([text])[:, 1][0])
    label = int(proba >= thr)
    label_name = "AI" if label == 1 else "Human"

    out = {
        "content_type": resolved_type,
        "model_path": str(model_path),
        "eval_summary_path": str(eval_summary_path),
        "mode": args.mode,
        "threshold": thr,
        "prob_ai": round(proba, 6),
        "pred_label": label_name
    }

    if args.show_prob:
        print(json.dumps(out, indent=2))
    else:
        print(label_name)

if __name__ == "__main__":
    main()
