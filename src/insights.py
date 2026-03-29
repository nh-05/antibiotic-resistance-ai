import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import pandas as pd
from src.config import AB_COLS


def get_drug_resistance(df):
    """
    Calculate resistance rate for each antibiotic
    """
    resistance = {}

    for col in AB_COLS:
        if col in df.columns:
            resistance[col] = (df[col] == "R").mean()

    return pd.Series(resistance).sort_values(ascending=False)


def get_drug_effectiveness(df):
    """
    Calculate susceptibility rate (S) for each antibiotic
    """
    effectiveness = {}

    for col in AB_COLS:
        if col in df.columns:
            effectiveness[col] = (df[col] == "S").mean()

    return pd.Series(effectiveness).sort_values(ascending=False)


def get_bacteria_resistance(df):
    """
    Calculate average resistance per bacteria
    """
    if "Species_clean" not in df.columns:
        return None

    bacteria_res = df.groupby("Species_clean")[AB_COLS].apply(
        lambda x: (x == "R").mean().mean()
    )

    return bacteria_res.sort_values(ascending=False)


def print_insights(df):
    """
    Print simple human-readable insights
    """

    print("\n📝 REAL-WORLD INSIGHTS:\n")

    # 1. Most resistant drugs
    drug_res = get_drug_resistance(df)
    print("🔴 Most Resistant Antibiotics:")
    for drug, rate in drug_res.head(5).items():
        print(f"- {drug} has high resistance ({rate*100:.1f}%)")

    # 2. Most effective drugs
    drug_eff = get_drug_effectiveness(df)
    print("\n🟢 Most Effective Antibiotics:")
    for drug, rate in drug_eff.head(5).items():
        print(f"- {drug} is highly effective ({rate*100:.1f}% susceptible)")

    # 3. Most resistant bacteria
    bacteria_res = get_bacteria_resistance(df)
    if bacteria_res is not None:
        print("\n🦠 Most Resistant Bacteria:")
        for bact, rate in bacteria_res.head(5).items():
            print(f"- {bact} shows high resistance ({rate*100:.1f}%)")

    print("\n💡 Summary:")
    print("- Some antibiotics are becoming ineffective due to high resistance.")
    print("- Certain bacteria show strong resistance patterns.")
    print("- Choosing the right antibiotic is critical for treatment success.")