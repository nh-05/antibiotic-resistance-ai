import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import pandas as pd
from typing import Dict
from src.config import AB_COLS


def compare_drugs(df_primary: pd.DataFrame, bacteria_name: str) -> Dict[str, float]:
    """
    Compare resistance of all antibiotics for a given bacteria.

    Parameters:
        df_primary (DataFrame): cleaned primary dataset
        bacteria_name (str): bacteria species name

    Returns:
        Dict[str, float]: {antibiotic: resistance_percentage}
    """

    # Filter dataset for selected bacteria
    df_bacteria = df_primary[df_primary["Species_clean"] == bacteria_name]

    if df_bacteria.empty:
        print(f"No data found for {bacteria_name}")
        return {}

    # Calculate resistance for each antibiotic
    resistance_rates = {}

    for ab in AB_COLS:
        if ab in df_bacteria.columns:
            total = df_bacteria[ab].notna().sum()
            resistant = (df_bacteria[ab] == "R").sum()

            if total > 0:
                resistance_rates[ab] = round((resistant / total) * 100, 2)
            else:
                resistance_rates[ab] = 0.0

    # Sort by highest resistance
    resistance_rates = dict(sorted(resistance_rates.items(), key=lambda x: x[1], reverse=True))

    return resistance_rates


def print_comparison(df_primary: pd.DataFrame, bacteria_name: str):
    """
    Print drug comparison nicely (for terminal/demo)
    """

    results = compare_drugs(df_primary, bacteria_name)

    if not results:
        return

    print(f"\n🧪 Drug Comparison for {bacteria_name}:\n")

    for drug, rate in results.items():
        print(f"{drug:20} → {rate:.2f}% resistant")

    # Highlight best & worst drugs
    best_drug = min(results, key=results.get)
    worst_drug = max(results, key=results.get)

    print("\n✅ Best Option (Lowest Resistance):")
    print(f"→ {best_drug} ({results[best_drug]:.2f}%)")

    print("\n⚠️ Worst Option (Highest Resistance):")
    print(f"→ {worst_drug} ({results[worst_drug]:.2f}%)")