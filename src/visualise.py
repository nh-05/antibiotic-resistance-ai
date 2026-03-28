import matplotlib
matplotlib.use("Agg")  # no pop-up windows, runs straight through
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import AB_COLS, CLSI_BREAKPOINTS, FIGURES_DIR


def plot_resistance_rates(df):
    resist_rates = {}
    for col in AB_COLS:
        total = df[col].notna().sum()
        r_count = (df[col] == "R").sum()
        resist_rates[col] = round(r_count / total * 100, 1) if total else 0

    resist_df = pd.Series(resist_rates).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#c04828" if r > 50 else "#BA7517" if r > 20 else "#3B8BD4"
              for r in resist_df]
    bars = ax.barh(resist_df.index, resist_df.values, color=colors, edgecolor="white")
    ax.set_xlabel("Resistance rate (%)")
    ax.set_title("Antibiotic Resistance Rates — Primary Dataset", fontweight="bold")
    ax.axvline(50, color="red", linestyle="--", alpha=0.4, label="50% threshold")
    ax.legend()
    for bar, val in zip(bars, resist_df.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_resistance_rates.png", dpi=150)
    plt.close()
    print("    Saved 01_resistance_rates.png")


def plot_species_heatmap(df):
    top_species = df["Species_clean"].value_counts().head(8).index.tolist()
    df_top = df[df["Species_clean"].isin(top_species)]

    hm_data = {}
    for sp in top_species:
        sub = df_top[df_top["Species_clean"] == sp]
        row = {}
        for ab in AB_COLS:
            total = sub[ab].notna().sum()
            row[ab] = round((sub[ab] == "R").sum() / total * 100, 1) if total else 0
        hm_data[sp] = row

    hm_df = pd.DataFrame(hm_data).T

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(hm_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, linewidths=0.5, ax=ax,
                cbar_kws={"label": "% Resistant"})
    ax.set_title("Resistance Rate (%) by Species × Antibiotic", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_species_antibiotic_heatmap.png", dpi=150)
    plt.close()
    print("    Saved 02_species_antibiotic_heatmap.png")


def plot_zone_distributions(df2):
    ab_to_plot = [ab for ab in CLSI_BREAKPOINTS if ab in df2.columns]
    n = len(ab_to_plot)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, ab in enumerate(ab_to_plot):
        bp = CLSI_BREAKPOINTS[ab]
        ax = axes[i]

        # Force numeric, keep zeros, only drop NaN
        col_data = pd.to_numeric(df2[ab], errors="coerce")
        col_data = col_data[col_data.notna()]  # keep 0s, drop only NaN

        ax.hist(col_data, bins=15, color="#3B8BD4", alpha=0.75, edgecolor="white")
        ax.axvline(bp["R"], color="#c04828", linestyle="--", linewidth=1.5, label=f"R≤{bp['R']}")
        ax.axvline(bp["S"], color="#639922", linestyle="--", linewidth=1.5, label=f"S≥{bp['S']}")
        ax.set_title(ab, fontsize=10, fontweight="bold")
        ax.set_xlabel("Zone (mm)")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7)

    fig.suptitle("Zone of Inhibition Distributions — Secondary Dataset", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_zone_distributions.png", dpi=150)
    plt.close()
    print("    Saved 03_zone_distributions.png")


def plot_mdr_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mdr_rate = df["is_MDR"].mean() * 100
    top_species = df["Species_clean"].value_counts().head(8).index.tolist()

    mdr_vals = df["resistance_count"].value_counts().sort_index()
    axes[0].bar(mdr_vals.index, mdr_vals.values,
                color=["#c04828" if i >= 3 else "#3B8BD4" for i in mdr_vals.index])
    axes[0].axvline(2.5, color="black", linestyle="--", alpha=0.5, label="MDR threshold")
    axes[0].set_xlabel("Number of resistant antibiotics")
    axes[0].set_ylabel("Isolate count")
    axes[0].set_title("Resistance Count Distribution")
    axes[0].legend()

    species_mdr = df.groupby("Species_clean")["is_MDR"].mean().loc[
        [s for s in top_species if s in df["Species_clean"].values]
    ] * 100
    species_mdr.sort_values().plot.barh(ax=axes[1], color="#c04828", alpha=0.8)
    axes[1].set_xlabel("MDR rate (%)")
    axes[1].set_title("MDR Rate by Top Species")
    axes[1].axvline(mdr_rate, color="navy", linestyle="--", alpha=0.4,
                    label=f"Overall {mdr_rate:.0f}%")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_mdr_analysis.png", dpi=150)
    plt.close()
    print("    Saved 04_mdr_analysis.png")


def plot_all(df_primary, df_secondary):
    plot_resistance_rates(df_primary)
    plot_species_heatmap(df_primary)
    plot_zone_distributions(df_secondary)
    plot_mdr_analysis(df_primary)