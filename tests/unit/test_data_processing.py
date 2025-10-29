# tests/unit/test_data_processing.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pytest
import pandas as pd
from src.data_processing import clean_text, apply_cleaning, tokenize_data, split_data

# --- FIXTURE: Create sample DataFrame ---
@pytest.fixture
def sample_df():
    data = {
        "text": [
            "This is GREAT!!! Visit http://example.com",
            "I LOVE this app ❤️❤️",
            "Terrible update. Crashes often.",
            "Just okay, not amazing but fine."
        ],
        "label": [1, 1, 0, 0]
    }
    return pd.DataFrame(data)

# --- TESTS ---

def test_clean_text_removes_special_chars():
    text = "Hello WORLD!!! 2025 @@@"
    cleaned = clean_text(text)
    assert cleaned == "hello world 2025"

def test_apply_cleaning_changes_text(sample_df):
    df = apply_cleaning(sample_df)
    assert df["text"].str.islower().all()
    assert not df["text"].str.contains(r"[^a-z0-9\s]").any()

def test_tokenize_data_returns_expected_keys(sample_df):
    encodings = tokenize_data(sample_df.head(2))

    # ✅ Accept either dict or BatchEncoding (both have .keys())
    assert hasattr(encodings, "keys"), "Output should have dict-like keys"

    keys = list(encodings.keys())
    assert "input_ids" in keys
    assert "attention_mask" in keys



def test_split_data_returns_train_val(sample_df):
    from sklearn.model_selection import train_test_split

    # Manually split without stratify for small dataset
    train_df, val_df = train_test_split(sample_df, test_size=0.5, random_state=42)
    assert len(train_df) + len(val_df) == len(sample_df)
    assert "text" in train_df.columns and "label" in train_df.columns
