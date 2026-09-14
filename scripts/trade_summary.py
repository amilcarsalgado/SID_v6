"""
===============================================================================
SCRIPT: COMPREHENSIVE TRADE PERFORMANCE SUMMARY
LOCATION: scripts/trade_summary.py
DESCRIPTION: Ingests the Alpaca_Live_Ledger.xlsx file and generates a detailed
             metrics breakdown including win rate, profit factor, and expectancy.
===============================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_trade_summary():
    # Point directly to the SIM_Results folder
    input_file = PROJECT_ROOT / "SIM_Results" / "Alpaca_Live_Ledger.xlsx"

    print(f"📊 Reading ledger from: {input_file}")

    try:
        # Load the closed trades
        df = pd.read_excel(input_file, sheet_name="Closed_Trades")
    except Exception as e:
        print(f"❌ Error loading ledger. Make sure the file exists: {e}")
        return

    if df.empty:
        print("⚠️ The ledger is empty. No trades to summarize.")
        return


    # Convert timestamps for duration calculations
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='mixed', utc=True)
    df['Exit_Time'] = pd.to_datetime(df['Exit_Time'], format='mixed', utc=True)
    df['Hold_Time_Hours'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds() / 3600

    # Categorize trades
    winning_trades = df[df['Realized_PnL'] > 0]
    losing_trades = df[df['Realized_PnL'] <= 0]

    # Core Metrics Calculations
    total_trades = len(df)
    wins = len(winning_trades)
    losses = len(losing_trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    gross_profit = winning_trades['Realized_PnL'].sum()
    gross_loss = abs(losing_trades['Realized_PnL'].sum())
    net_pnl = df['Realized_PnL'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    avg_win = winning_trades['Realized_PnL'].mean() if wins > 0 else 0.0
    avg_loss = losing_trades['Realized_PnL'].mean() if losses > 0 else 0.0
    biggest_win = df['Realized_PnL'].max() if wins > 0 else 0.0
    biggest_loss = df['Realized_PnL'].min() if losses > 0 else 0.0

    avg_return_pct = df['Return_%'].mean()
    avg_hold_time = df['Hold_Time_Hours'].mean()
    trading_days = df['Exit_Time'].dt.date.nunique()

    # Handle best ticker safely
    best_ticker = "N/A"
    if wins > 0:
        best_ticker = df.groupby('Ticker')['Realized_PnL'].sum().idxmax()

    # Compile Summary Data
    summary_data = {
        "Metric": [
            "Total Trades", "Winning Trades", "Losing Trades", "Win Rate",
            "Net Realized PnL", "Gross Profit", "Gross Loss", "Profit Factor",
            "Average Return %", "Average Win", "Average Loss",
            "Biggest Win", "Biggest Loss", "Average Hold Time (Hours)",
            "Active Trading Days", "Best Performing Ticker"
        ],
        "Value": [
            total_trades, wins, losses, win_rate,
            net_pnl, gross_profit, gross_loss, profit_factor,
            avg_return_pct, avg_win, avg_loss,
            biggest_win, biggest_loss, avg_hold_time,
            trading_days, best_ticker
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Dynamic File Naming inside SIM_Results (Format: Trade_Summary_ddmmmyy.xlsx)
    date_str = datetime.now().strftime("%d%b%y").lower()
    output_filename = PROJECT_ROOT / "SIM_Results" / f"Trade_Summary_{date_str}.xlsx"

    # Export with formatted columns
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name="Performance_Summary", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Performance_Summary"]

        # Formatting objects
        currency_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        num_fmt = workbook.add_format({'num_format': '0.00'})
        bold_fmt = workbook.add_format({'bold': True})

        worksheet.set_column('A:A', 30, bold_fmt)
        worksheet.set_column('B:B', 20)

        # Apply specific formats to value cells based on the metric type
        for row_num, metric in enumerate(summary_df['Metric']):
            val = summary_df.iloc[row_num]['Value']
            cell = f'B{row_num + 2}'

            if pd.isna(val):
                worksheet.write(cell, "N/A")
                continue

            if "PnL" in metric or "Profit" in metric or "Loss" in metric or "Win" in metric and "Rate" not in metric and "Trades" not in metric:
                if isinstance(val, (int, float)):
                    worksheet.write(cell, val, currency_fmt)
            elif "Rate" in metric:
                worksheet.write(cell, val, pct_fmt)
            elif "Factor" in metric or "Time" in metric or "Return" in metric:
                if isinstance(val, (int, float)):
                    worksheet.write(cell, val, num_fmt)
            else:
                worksheet.write(cell, val)

    print(f"✅ Comprehensive trade summary exported to: {output_filename}")


if __name__ == "__main__":
    generate_trade_summary()