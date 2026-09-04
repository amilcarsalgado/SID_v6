"""
===============================================================================
SCRIPT: HOURLY ML DATASET GENERATOR & GHOST EXIT ORACLE
LOCATION: scripts/build_hourly_dataset.py
===============================================================================
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_engine.data.loader import MarketDataLoader

REVISION = "2.0_Hourly"
RSI_TARGETS = [50, 55, 60, 65, 70]


def generate_hourly_ghost_exits(future_df: pd.DataFrame, entry_price: float, stop_loss_price: float) -> dict:
    """
    Looks ahead across hourly price action to calculate MFE,
    Stop Loss Death Timestamps, and Target Plateaus at RSI [50, 55, 60, 65, 70].
    """
    results = {
        'MFE_Price': entry_price,
        'MFE_PnL': 0.0,
        'Stop_Loss_Hit_Time': 'Never',
        'Bars_To_Death': len(future_df),
        'Target_Label': 0  # 0: Trap, 1: Base Hit (RSI 50), 2: Whale (RSI 60)
    }

    for target in RSI_TARGETS:
        results[f'Price_at_RSI_{target}'] = stop_loss_price
        results[f'PnL_at_RSI_{target}'] = round(stop_loss_price - entry_price, 2)
        results[f'Hit_RSI_{target}'] = 0

    if future_df.empty:
        return results

    # 1. Detect Stop Loss Death Bar
    stop_hit_condition = future_df['Low'] <= stop_loss_price
    if stop_hit_condition.any():
        death_time = stop_hit_condition.idxmax()
        lifespan_df = future_df.loc[:death_time]
        results['Stop_Loss_Hit_Time'] = str(death_time)
        results['Bars_To_Death'] = len(lifespan_df)
    else:
        lifespan_df = future_df

    # 2. Maximum Favorable Excursion (MFE) during lifespan
    if not lifespan_df.empty:
        max_high = lifespan_df['High'].max()
        if max_high > entry_price:
            results['MFE_Price'] = round(max_high, 2)
            results['MFE_PnL'] = round(max_high - entry_price, 2)

    # 3. Check RSI Target Hits before Stop Loss
    for target in RSI_TARGETS:
        target_hit = lifespan_df[lifespan_df['RSI'] >= target]
        if not target_hit.empty:
            hit_time = target_hit.index[0]
            hit_price = lifespan_df.loc[hit_time, 'Close']
            results[f'Price_at_RSI_{target}'] = round(hit_price, 2)
            results[f'PnL_at_RSI_{target}'] = round(hit_price - entry_price, 2)
            results[f'Hit_RSI_{target}'] = 1

    # 4. Multi-Class Label Assignment
    if results['Hit_RSI_60'] == 1 and results['PnL_at_RSI_60'] > 0:
        results['Target_Label'] = 2
    elif results['Hit_RSI_50'] == 1 and results['PnL_at_RSI_50'] > 0:
        results['Target_Label'] = 1
    else:
        results['Target_Label'] = 0

    return results


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 65)
    print("🧠 HOURLY ML TRAINING DATASET & GHOST EXIT GENERATOR")
    print("=" * 65)

    config = load_settings()
    loader = MarketDataLoader(config)

    dataset_dir = PROJECT_ROOT / "data" / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sim_results_dir = PROJECT_ROOT / "SIM_Results"
    sim_results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sift Watchlist
    tickers, sector_map, earnings_df = loader.load_watchlist(min_score=1)
    print(f"📋 Loaded {len(tickers)} scored tickers for dataset mining.")

    # 2. Load Cached 1-Hour Bar Data
    t0 = time.time()
    df_bars = loader.fetch_historical_bars(tickers=tickers, force_refresh=False)
    print(f"⚡ Loaded {len(df_bars):,} hourly candles from cache in {time.time() - t0:.2f}s")

    # 3. Vectorized Indicator Engine per Ticker
    print("\n⚙️ Calculating Technical Indicators across all hourly candles...")
    all_processed = []
    unique_tickers = df_bars.index.get_level_values("Ticker").unique()

    for i, t in enumerate(unique_tickers, 1):
        df = df_bars.xs(t, level='Ticker').copy()
        df.dropna(how='all', inplace=True)
        if len(df) < 250:
            continue

        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()

        tr = np.maximum((df['High'] - df['Low']),
                        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = tr.rolling(14).mean()

        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))

        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['R_Hook_U'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) <= df['RSI'].shift(2))
        df['M_Hook_U'] = (df['MACD'] > df['MACD'].shift(1)) & (df['MACD'].shift(1) <= df['MACD'].shift(2))

        df['State_L_Agg'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] >= 50, 0, np.nan))
        df['State_L_Tier'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] >= 50, 0, np.nan))
        df['State_L_Agg'] = df['State_L_Agg'].ffill().fillna(0)
        df['State_L_Tier'] = df['State_L_Tier'].ffill().fillna(0)
        df['L_ID'] = ((df['State_L_Tier'] == 1) & (df['State_L_Tier'].shift(1) == 0)).cumsum()

        valid_l = df['Low'].where(df['State_L_Tier'] == 1, np.nan)
        df['Lowest_Since'] = valid_l.groupby(df['L_ID']).cummin()
        valid_rsi = df['RSI'].where(df['State_L_Tier'] == 1, np.nan)
        df['Extreme_RSI'] = valid_rsi.groupby(df['L_ID']).cummin()
        df['Extreme_Date'] = df.index.to_series().where(df['State_L_Tier'] == 1, pd.NaT).groupby(df['L_ID']).transform('first')
        df['SL_L'] = np.floor(df['Lowest_Since'])

        df['L_Agg_Sig'] = (df['State_L_Agg'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (df['Close'] > df['EMA_200'])
        df['L_Tier_Sig'] = (df['State_L_Tier'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (df['Close'] > df['EMA_200'])

        all_processed.append((t, df))
        if i % 100 == 0 or i == len(unique_tickers):
            print(f"  • Processed indicators for {i}/{len(unique_tickers)} tickers...", flush=True)

    # 4. Mine Setups and Compute Forward Ghost Exits
    print("\n🔭 Mining trade candidate setups and forward Ghost Exits...")
    records = []
    obs_start = pd.to_datetime("2021-06-01").tz_localize("America/New_York")

    # Baseline Portfolio Simulation Metrics for the Summary Page
    initial_balance = 10000.0
    balance = initial_balance
    current_equity = initial_balance
    commission = 1.00

    for t, df in all_processed:
        t_earnings = earnings_df[earnings_df['Ticker'] == t]['Earnings_Date']
        signal_mask = (df.index >= obs_start) & (df['L_Agg_Sig'] | df['L_Tier_Sig'])
        signal_indices = df.index[signal_mask]

        for dt in signal_indices:
            row = df.loc[dt]
            entry_price = row['Close']
            stop_loss = row['SL_L']

            if pd.isna(entry_price) or pd.isna(stop_loss) or entry_price <= stop_loss:
                continue

            applied_risk_pct = 0.02 if row['L_Agg_Sig'] else 0.005
            risk_tier_str = "2.0%" if row['L_Agg_Sig'] else "0.5%"

            # Calculate Earnings Proximity
            days_to_earnings, days_since_earnings = 45, 45
            if not t_earnings.empty:
                deltas = (t_earnings.dt.tz_localize(dt.tz) - dt).dt.days
                future_deltas = deltas[deltas >= 0]
                past_deltas = deltas[deltas < 0]
                if not future_deltas.empty: days_to_earnings = int(future_deltas.min())
                if not past_deltas.empty: days_since_earnings = int(abs(past_deltas.max()))

            # Slice Forward Lifespan
            future_df = df.loc[df.index > dt]
            ghost_exits = generate_hourly_ghost_exits(future_df, entry_price, stop_loss)

            # Sizing and PnL calculation for baseline RSI 50 Target exit
            dollar_risk = current_equity * applied_risk_pct
            risk_per_share = abs(entry_price - stop_loss)
            shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

            if ghost_exits['Hit_RSI_50'] == 1 and ghost_exits['PnL_at_RSI_50'] > 0:
                exit_price = ghost_exits['Price_at_RSI_50']
                exit_reason = "RSI 50 Target"
            else:
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"

            total_profit = (shares * (exit_price - entry_price)) - (commission * 2) if shares > 0 else 0.0
            current_equity += total_profit

            feature_row = {
                'Ticker': t,
                'Sector_Name': sector_map.get(t, 'Unmapped'),
                'Direction': 'LONG',
                'Applied_Risk_Pct': risk_tier_str,
                'RSI_Extreme_Value': round(row['Extreme_RSI'], 2),
                'Entry_Date': str(dt.date()),
                'Entry_Price': round(entry_price, 2),
                'Entry_EMA_200': round(row['EMA_200'], 2),
                'Dist_From_EMA_%': round(((entry_price - row['EMA_200']) / row['EMA_200']) * 100, 2),
                'Entry_RVOL': round(row['RVOL'], 2),
                'Entry_ATR': round(row['ATR'], 2),
                'Exit_Date': str(ghost_exits['Stop_Loss_Hit_Time']).split(" ")[0] if ghost_exits['Stop_Loss_Hit_Time'] != 'Never' else str(dt.date()),
                'Hold_Duration': ghost_exits['Bars_To_Death'],
                'Exit_Price': round(exit_price, 2),
                'Exit_RSI': 50.0 if exit_reason == "RSI 50 Target" else round(row['RSI'], 2),
                'Exit_Reason': exit_reason,
                'Net_PnL (After Fees)': round(total_profit, 2),
                'Account_Equity': round(current_equity, 2),
                'MFE_Price': ghost_exits['MFE_Price'],
                'MFE_PnL': ghost_exits['MFE_PnL'],
                'Stop_Loss_Hit_Date': ghost_exits['Stop_Loss_Hit_Time'],
                'Price_at_RSI_50': ghost_exits['Price_at_RSI_50'],
                'PnL_at_RSI_50': ghost_exits['PnL_at_RSI_50'],
                'Price_at_RSI_55': ghost_exits['Price_at_RSI_55'],
                'PnL_at_RSI_55': ghost_exits['PnL_at_RSI_55'],
                'Price_at_RSI_60': ghost_exits['Price_at_RSI_60'],
                'PnL_at_RSI_60': ghost_exits['PnL_at_RSI_60'],
                'Price_at_RSI_65': ghost_exits['Price_at_RSI_65'],
                'PnL_at_RSI_65': ghost_exits['PnL_at_RSI_65'],
                'Price_at_RSI_70': ghost_exits['Price_at_RSI_70'],
                'PnL_at_RSI_70': ghost_exits['PnL_at_RSI_70'],
                'Days_To_Earnings': days_to_earnings,
                'Days_Since_Earnings': days_since_earnings,
                'Entry_Month': dt.month,
                'Risk_Tier': applied_risk_pct,
                'Target_Label': ghost_exits['Target_Label']
            }
            records.append(feature_row)

    dataset_df = pd.DataFrame(records).sort_values(by=['Entry_Date', 'Ticker'])
    print(f"\n📊 Extraction Complete: Mined {len(dataset_df):,} total hourly candidate trade setups.")

    # 5. Build Summary Sheet Data (Exact Legacy v1.31 Format)
    total_trades = len(dataset_df)
    wins = dataset_df[dataset_df['Net_PnL (After Fees)'] > 0]
    losses = dataset_df[dataset_df['Net_PnL (After Fees)'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = wins['Net_PnL (After Fees)'].sum()
    gross_loss = abs(losses['Net_PnL (After Fees)'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    unique_dates = dataset_df['Entry_Date'].unique()
    total_return = ((current_equity - initial_balance) / initial_balance) * 100

    summary_data = {
        "Metric": [
            "Simulation Revision Level",
            "Total Trading Days Processed",
            "Total Duration (Months)",
            "Initial Account Value",
            "Final Account Equity (Cash + Invested)",
            "Available Cash",
            "Capital Still Invested",
            "Total Return (%)",
            "Total Closed Trades",
            "Winning Trades",
            "Losing Trades",
            "Win Rate (%)",
            "Profit Factor (Gross Profit / Gross Loss)",
            "Biggest Win",
            "Biggest Loss"
        ],
        "Value": [
            REVISION,
            len(unique_dates),
            round(len(unique_dates) / 21, 1),
            f"${initial_balance:,.2f}",
            f"${current_equity:,.2f}",
            f"${current_equity:,.2f}",
            "$0.00",
            f"{total_return:.2f}%",
            total_trades,
            len(wins),
            len(losses),
            f"{win_rate:.2f}%",
            f"{profit_factor:.2f}",
            f"${wins['Net_PnL (After Fees)'].max():,.2f}" if not wins.empty else "N/A",
            f"${losses['Net_PnL (After Fees)'].min():,.2f}" if not losses.empty else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # 6. Export Parquet, CSV, and Excel
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = dataset_dir / "hourly_ml_training_data.parquet"
    csv_path = dataset_dir / "hourly_ml_training_data.csv"
    excel_path = sim_results_dir / f"Simulation_Results_{REVISION}_{ts}.xlsx"

    print("\n💾 Saving Dataset Artifacts...")

    try:
        dataset_df.to_parquet(parquet_path, index=False)
        print(f"  📁 Parquet : {parquet_path}")
    except Exception as e:
        print(f"  ⚠️ Parquet export skipped ({e}). CSV and Excel saved successfully.")

    dataset_df.to_csv(csv_path, index=False)
    print(f"  📁 CSV     : {csv_path}")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        dataset_df.to_excel(writer, sheet_name='Master Trading Log', index=False)
    print(f"  📁 Excel   : {excel_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()