import numpy as np
from src.config import AB_COLS


def add_mdr_flags(df):
    ab_enc = df[AB_COLS].map(
        lambda x: 1 if x == "R" else 0.5 if x == "I" else 0 if x == "S" else np.nan
    )
    df["resistance_count"] = ab_enc.apply(lambda row: (row == 1).sum(), axis=1)
    df["is_MDR"] = (df["resistance_count"] >= 3).astype(int)
    return df


def encode_clinical(df):
    return df