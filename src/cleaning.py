import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import AB_COLS, CLSI_BREAKPOINTS, SPECIES_MAP


def normalise_resistance(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().upper()
    if v in {"R", "RESISTANT"}:
        return "R"
    if v in {"S", "SUSCEPTIBLE"}:
        return "S"
    if v in {"I", "INTERMEDIATE"}:
        return "I"
    return np.nan


def clean_species(s):
    if pd.isna(s):
        return np.nan
    raw = str(s).strip()
    parts = raw.split(" ", 1)
    species = parts[1].strip() if len(parts) == 2 and parts[0].startswith("S") else raw
    return SPECIES_MAP.get(species.upper(), species)


def parse_age(ag):
    if pd.isna(ag):
        return np.nan, np.nan
    s = str(ag).strip()
    gender = np.nan
    if s.upper().endswith("M"):
        gender = "M"
        s = s[:-1]
    elif s.upper().endswith("F"):
        gender = "F"
        s = s[:-1]
    try:
        return float(s), gender
    except ValueError:
        return np.nan, gender


def encode_binary(col):
    mapping = {"YES": 1, "OUI": 1, "1": 1, "NO": 0, "NON": 0, "0": 0}
    return col.astype(str).str.upper().str.strip().map(mapping)


def classify_zone(val, ab):
    if pd.isna(val):
        return np.nan
    bp = CLSI_BREAKPOINTS[ab]
    if val <= bp["R"]:
        return "R"
    if val >= bp["S"]:
        return "S"
    return "I"


def clean_primary(df):
    # Normalise antibiotic columns
    for col in AB_COLS:
        df[col] = df[col].apply(normalise_resistance)

    # Clean species names
    df["Species_clean"] = df["Souches"].apply(clean_species)

    # Parse age and gender
    df[["Age", "Gender"]] = df["age/gender"].apply(
        lambda x: pd.Series(parse_age(x))
    )

    # Encode clinical features
    for feat in ["Diabetes", "Hypertension", "Hospital_before"]:
        df[feat + "_enc"] = encode_binary(df[feat])

    # Parse dates — format="mixed" handles inconsistent date formats without warning
    df["Date_clean"] = pd.to_datetime(
        df["Collection_Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )
    df["Year"] = df["Date_clean"].dt.year

    return df


def clean_secondary(df):
    for ab in CLSI_BREAKPOINTS:
        df[ab + "_class"] = df[ab].apply(lambda v: classify_zone(v, ab))

    le = LabelEncoder()
    df["Location_enc"] = le.fit_transform(df["Location"].astype(str))

    return df