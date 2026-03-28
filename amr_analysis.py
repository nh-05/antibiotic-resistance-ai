import pandas as pd

print("Step 1 — Loading datasets...")
from src.config import PRIMARY_CSV, SECONDARY_XLSX, DATA_PROCESSED
df_primary   = pd.read_csv(PRIMARY_CSV)
df_secondary = pd.read_excel(SECONDARY_XLSX)
print("Done loading")

print("Step 2 — Cleaning...")
from src.cleaning import clean_primary, clean_secondary
df_primary   = clean_primary(df_primary)
df_secondary = clean_secondary(df_secondary)
print("Done cleaning")

print("Step 3 — Features...")
from src.features import add_mdr_flags, encode_clinical
df_primary = add_mdr_flags(df_primary)
df_primary = encode_clinical(df_primary)
print("Done features")

print("Step 4 — Saving cleaned data...")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
df_primary.to_csv(DATA_PROCESSED / "primary_cleaned.csv", index=False)
df_secondary.to_csv(DATA_PROCESSED / "secondary_cleaned.csv", index=False)
print("Cleaned data saved to data/processed/")

print("Step 5 — Visualising...")
from src.visualise import plot_all
plot_all(df_primary, df_secondary)
print("Done visualising")

print("Step 6 — Training models...")
from src.models import run_all_models
run_all_models(df_primary, df_secondary)
print("Done models")

print("\nAll done! Check outputs/figures/ and outputs/models/")