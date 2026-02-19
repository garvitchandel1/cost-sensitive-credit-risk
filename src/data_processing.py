
import pandas as pd
from sklearn.model_selection import train_test_split


def drop_leakage_columns(df):
    leakage_cols = [
        'total_rec_int',
        'recoveries',
        'collection_recovery_fee',
        'last_pymnt_d',
        'last_pymnt_amnt'
    ]

    existing = [col for col in leakage_cols if col in df.columns]
    return df.drop(columns=existing)


def prepare_features(df, target_column='target'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )