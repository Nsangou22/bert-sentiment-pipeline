# src/data_processing.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters, URLs, and extra spaces.
    Returns the lowercase version of the text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s]", "", text)  # keep only alphanumeric and spaces
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_text() to the 'text' column of the DataFrame.
    """
    if "text" not in df.columns:
        raise ValueError("❌ 'text' column not found in the DataFrame.")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    return df


def tokenize_data(df: pd.DataFrame, tokenizer_name: str = "bert-base-uncased"):
    """
    Tokenizes the text data using the specified Hugging Face tokenizer.
    Returns a dictionary of tokenized sequences.
    """
    if "text" not in df.columns:
        raise ValueError("❌ 'text' column not found in the DataFrame for tokenization.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )
    print(f"✅ Tokenization complete! Tokens: {len(encodings['input_ids'])}")
    return encodings


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Splits the dataset into train and validation sets.
    Returns train_df, val_df.
    """
    if "label" not in df.columns:
        raise ValueError("❌ 'label' column not found in the DataFrame.")
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    print(f"✅ Data split complete! Train size: {len(train_df)}, Val size: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
