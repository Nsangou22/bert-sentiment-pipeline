# tests/unit/test_data_extraction.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import pandas as pd
from src.data_extraction import load_data
from pathlib import Path

def test_load_data_success():
    """✅ Should load dataset successfully and return DataFrame."""
    df = load_data("data/dataset.csv")
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0  # not empty

def test_missing_file():
    """❌ Should raise ValueError when file is missing."""
    with pytest.raises(ValueError):
        load_data("data/non_existent.csv")

def test_missing_columns(tmp_path):
    """❌ Should raise ValueError when columns are wrong."""
    # create fake csv missing 'content'/'score'
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("review,sentiment\nhello,1")
    with pytest.raises(ValueError):
        load_data(str(bad_csv))
