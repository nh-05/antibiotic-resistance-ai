import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "primary_cleaned.csv"

df = pd.read_csv(DATA_PATH)

AB_COLS = [
    "AMC", "CTX/CRO", "FOX", "IPM",
    "AMX/AMP", "CZ", "Furanes",
    "Co-trimoxazole", "colistine"
]

def recommend_antibiotics(bacteria_name, top_n=5):

    df_bac = df[df["Species_clean"] == bacteria_name]

    if df_bac.empty:
        return []

    resistance = {
        ab: (df_bac[ab] == "R").mean()
        for ab in AB_COLS
    }

    sorted_drugs = sorted(resistance.items(), key=lambda x: x[1])

    return sorted_drugs[:top_n]