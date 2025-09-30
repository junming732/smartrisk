from datasets import Dataset
import pandas as pd
from typing import List


def load_disaster_tweets(train_csv: str, text_col: str, label_col: str) -> Dataset:
    df = pd.read_csv(train_csv)
    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)


def load_toy_texts(texts: List[str], labels: List[int]) -> Dataset:
    df = pd.DataFrame({"text": texts, "label": labels})
    return Dataset.from_pandas(df, preserve_index=False)
