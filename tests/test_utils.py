import sys
from pathlib import Path

import pandas as pd

# Ensure root path is on sys.path to import src package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import basic_clean


def test_basic_clean_handles_none_text():
    df = pd.DataFrame({'text_content': [None], 'label': [1]})
    cleaned = basic_clean(df)
    assert cleaned.loc[0, 'text_content'] == ""


def test_basic_clean_casts_labels_to_int():
    df = pd.DataFrame({'text_content': ['hello', 'world'], 'label': ['1', 0.0]})
    cleaned = basic_clean(df)
    assert cleaned['label'].tolist() == [1, 0]
    assert pd.api.types.is_integer_dtype(cleaned['label'])
