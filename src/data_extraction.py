# src/data_extraction.py
import pandas as pd
from pathlib import Path

def load_data(path: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Loads the teacher's dataset (with 'content' and 'score' columns)
    and renames them to 'text' and 'label'.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"❌ File not found at {path}")

    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise ValueError(f"❌ Failed to read CSV file: {e}")

    required_cols = {"content", "score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Expected columns {required_cols}, found {list(df.columns)}")

    # ✅ rename for the rest of the pipeline
    df = df.rename(columns={"content": "text", "score": "label"})
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    return df
