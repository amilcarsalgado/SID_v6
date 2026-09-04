"""
===============================================================================
SCRIPT: ANALYZE DATASET BY TICKER SCORE TIERS
LOCATION: scripts/analyze_score_tiers.py
===============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load Mined Dataset
csv_path = PROJECT_ROOT / "data" / "datasets" / "hourly_ml_training_data.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_path}. Run build_hourly_dataset.py first.")

df_data = pd.read_csv(csv_path)

# 2. Load Watchlist Scores
symbols_path = PROJECT_ROOT / "config" / "symbols.xlsx"
wl_df = pd.read_excel(symbols_path, sheet_name="Main")
wl_df["Ticker"] = wl_df["Ticker"].astype(str).str.strip().str.upper()

# Target Column G (Score)
if "Score" in wl_df.columns:
    wl_df["Score"] = pd.to_numeric(wl_df["Score"], errors="coerce").fillna(0)
elif "Ticker_Score" in wl_df.columns:
    wl_df["Score"] = pd.to_numeric(wl_df["Ticker_Score"], errors="coerce").fillna(0)
else:
    wl_df["Score"] = pd.to_numeric(wl_df.iloc[:, 6], errors="coerce").fillna(0)

score_map = dict(zip(wl_df["Ticker"], wl_df["Score"]))
df_data["Ticker_Score"] = df_data["Ticker"].map(score_map).fillna(0)


# 3. Assign Tiers
def assign_tier(score):
    if score >= 6:
        return "Tier 1: High Conviction (6-10)"
    elif score >= 3:
        return "Tier 2: Core Watchlist (3-5)"
    else:
        return "Tier 3: Speculative (1-2)"


df_data["Score_Tier"] = df_data["Ticker_Score"].apply(assign_tier)

# 4. Compute Metrics per Tier
results = []
for tier, group in df_data.groupby("Score_Tier"):
    total_setups = len(group)
    wins = group[group["Net_PnL (After Fees)"] > 0]
    losses = group[group["Net_PnL (After Fees)"] <= 0]

    win_rate = (len(wins) / total_setups) * 100 if total_setups else 0
    gross_win = wins["Net_PnL (After Fees)"].sum()
    gross_loss = abs(losses["Net_PnL (After Fees)"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.nan

    # ML Class distribution
    traps = len(group[group["Target_Label"] == 0])
    whales = len(group[group["Target_Label"] == 2])

    results.append({
        "Tier Group": tier,
        "Unique Tickers": group["Ticker"].nunique(),
        "Total Setups": total_setups,
        "Win Rate (%)": f"{win_rate:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Avg MFE ($)": f"${group['MFE_PnL'].mean():.2f}",
        "Trap Rate (Class 0)": f"{(traps / total_setups) * 100:.1f}%",
        "Whale Rate (Class 2)": f"{(whales / total_setups) * 100:.1f}%"
    })

tier_summary = pd.DataFrame(results).sort_values("Tier Group")
print("\n" + "=" * 80)
print("📊 HOURLY DATASET BREAKDOWN BY TICKER SCORE TIERS")
print("=" * 80)
print(tier_summary.to_string(index=False))
print("=" * 80 + "\n")