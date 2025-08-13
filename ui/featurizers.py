# src/featurizers.py
from __future__ import annotations
import re
import numpy as np

def code_features(texts: list[str]) -> np.ndarray:
    """
    Cheap, language-agnostic code features:
    - comment ratio, symbol ratio, average line length, avg indent depth,
      camelCase vs snake_case ratio, digit ratio.
    """
    feats = []
    for t in texts:
        s = t or ""
        lines = s.splitlines() or [""]
        n = max(len(s), 1)
        n_lines = max(len(lines), 1)

        comment_like = sum(1 for L in lines if L.strip().startswith(("//", "#", "--", "/*", "*", ";")))
        comment_ratio = comment_like / n_lines

        symbols = sum(ch in "{}[]();,:<>+-=*/%&|^~" for ch in s)
        symbol_ratio = symbols / n

        avg_line_len = sum(len(L) for L in lines) / n_lines
        indent_depth = np.mean([len(L) - len(L.lstrip(" \t")) for L in lines])

        camel = len(re.findall(r"[a-z]+[A-Z][a-zA-Z0-9]*", s))
        snake = len(re.findall(r"[a-z]+_[a-z0-9_]+", s))
        camel_snake_ratio = camel / max(snake, 1)

        digits = sum(ch.isdigit() for ch in s)
        digit_ratio = digits / n

        feats.append([comment_ratio, symbol_ratio, avg_line_len, indent_depth, camel_snake_ratio, digit_ratio])

    return np.array(feats, dtype=float)
