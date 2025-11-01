import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1", sep=";")
    df = df.rename(columns={"req_text": "text", "profession": "label"})
    return df


def make_splits(
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int = 99
) -> DatasetDict:
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df["label"], random_state=seed)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    return dataset
