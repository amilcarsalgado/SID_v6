"""
===============================================================================
SCRIPT: UNIFIED ACCOUNT TELEMETRY & PERFORMANCE DASHBOARD
LOCATION: scripts/account_summary.py
AUTHOR: Alvaro Salgado
DESCRIPTION: Combines live Alpaca open position polling, dynamic RSI calculation,
             historical ledger ingestion, and Plotly dashboard generation.
             Outputs to console, a multi-tab Excel file, and an interactive HTML plot.
===============================================================================

===============================================================================
🤖 PERMANENT DIRECTIVE TO AI CO-PILOT & QUANTITATIVE SYSTEMS ENGINEERS:
-------------------------------------------------------------------------------
The engineering notes in this header and throughout this file are permanent
architectural records and system specifications.
THEY MUST NEVER BE REMOVED, TRUNCATED, SUMMARIZED, OR OMITTED in future code
refactors or session transfers unless a specific architectural component has
been completely deprecated and physically removed from the codebase.
From now on, all subsequent scripts and revisions must retain this engineering
dossier to prevent context drift and ensure zero-tolerance rule enforcement.
===============================================================================

ENGINEERING DOSSIER & ARCHITECTURAL SPECIFICATIONS (SID EQUITIES V7.0):
-------------------------------------------------------------------------------
1. Core Operating Cadence:
   - Operates on a 7-bar daily schedule using 60-minute timeframe bars.
   - Scan schedule: 10:30, 11:30, 12:30, 01:30, 02:30, 03:30, and 03:58 PM NY.
   - Scan 7 (03:58 PM Pre-Close) normalizes volume by pro-rating the 28 elapsed
     minutes across a standard 30-minute half-bar window (volume * 30.0 / 28.0).

2. Execution Hierarchy & Priority Queues (Audited by this script):
   - Priority 0: Stop-Loss Interceptor. Audits closed broker orders to identify
     positions liquidated between scans and appends them to Alpaca_Live_Ledger.xlsx.
   - Priority 1: Immediate Exits. Evaluates active positions for profit-taking
     targets (Standard: RSI >= 50, Whale: RSI >= 60). Unconditional execution.
   - Priority 2: Macro Blackout Gatekeeper. Audits UTC timestamps against
     config/blackout_dates.json. If active, vetoes entry scanning entirely.
   - Priority 3: Immediate Entries. Scans watchlist for technical hooks (200 EMA,
     RSI < 35/30, MACD up-hook), validates ML Oracle v6.0, and executes OTO orders.

3. Macro Blackout Architecture (config/blackout_dates.json):
   - Scope: Strictly halts *new entry execution*. It NEVER pauses take-profit
     monitoring, nor does it affect resting broker-side stop-losses.
   - The Pre-Event Liquidity Vacuum: Market makers pull liquidity 30-60 minutes
     prior to Tier-1 releases (FOMC, CPI, NFP). Spreads widen and technical
     indicators emit false "chop" hooks.
   - Asset Blast Radius: Evaluates the 'impacted_assets' array. Events tagged
     'EQUITIES' or 'USD' halt this daemon; localized currency tags ('EUR', 'GBP')
     are reserved for the Forex expansion.

4. Strict Cash Collar & Money Management:
   - Zero margin allowed. The engine physically checks `account.cash` before entry.
   - Sizing calculation: min(risk_shares, max_notional_shares, max_cash_shares).
   - Max portfolio exposure: 15 open positions. Max single asset: 20% equity.

5. Capital Defense & Governors (Tracked by this script):
   - Drawdown Governor: Continuously updates the All-Time High Water Mark (HWM).
     If portfolio equity drops >= 5% from HWM, shifts to 🛡️ DEFENSIVE mode,
     instantly slashing risk allocations in half (Aggressive: 1%, Tier: 0.25%).
   - Regime Burn Guard: Once a trade closes (stop or profit), its unique setup
     regime ID (L_ID) is written to alpaca_state.json to prevent revenge trading.

6. ML Telemetry & Ghost Ledger (Oracle v7 Pipeline):
   - Every evaluated setup (both executed BOUGHT and vetoed TRAP setups) writes
     a full feature snapshot to data/training_store/oracle_v7_live_training_store.parquet.
   - Preserves un-executed setups to eliminate survivorship bias and train Oracle v7.

7. Known System Vulnerabilities & Fallback Roadmap:
   - Free Alpaca IEX data feed is prone to volume dropouts. When a bar query fails
     or drops volume, the engine must gracefully log the failure. A secondary
     fallback decorator (e.g., yfinance) is queued for future hardening.
===============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import zoneinfo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def generate_performance_summary(ledger_path):
    """Parses the closed trades ledger and calculates quantitative metrics."""
    try:
        df = pd.read_excel(ledger_path, sheet_name="Closed_Trades")
    except Exception as e:
        return None, f"Error loading ledger: {e}"

    if df.empty:
        return None, "Ledger is empty."

    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='mixed', utc=True)
    df['Exit_Time'] = pd.to_datetime(df['Exit_Time'], format='mixed', utc=True)
    df['Hold_Time_Hours'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds() / 3600

    winning_trades = df[df['Realized_PnL'] > 0]
    losing_trades = df[df['Realized_PnL'] <= 0]

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

    # Active vs Calendar Days
    trading_days = df['Exit_Time'].dt.date.nunique()
    first_trade = df['Entry_Time'].min()
    last_trade = df['Exit_Time'].max()
    calendar_days = (last_trade - first_trade).days + 1 if pd.notna(first_trade) else 0

    # Expectancy Calculation
    loss_rate = 1.0 - win_rate
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    best_ticker = df.groupby('Ticker')['Realized_PnL'].sum().idxmax() if wins > 0 else "N/A"

    summary_data = {
        "Metric": [
            "Total Trades", "Winning Trades", "Losing Trades", "Win Rate",
            "Net Realized PnL", "Gross Profit", "Gross Loss",
            "Profit Factor (Gross Profit/Gross Loss)", "Expectancy per Trade",
            "Average Return %", "Average Win", "Average Loss",
            "Biggest Win", "Biggest Loss", "Average Hold Time (Hours)",
            "Active Trading Days", "Total Calendar Days", "Best Performing Ticker"
        ],
        "Value": [
            total_trades, wins, losses, win_rate,
            net_pnl, gross_profit, gross_loss, profit_factor, expectancy,
            avg_return_pct, avg_win, avg_loss,
            biggest_win, biggest_loss, avg_hold_time,
            trading_days, calendar_days, best_ticker
        ]
    }
    return pd.DataFrame(summary_data), "Success"

def log_and_plot_performance_history(summary_df):
    """Appends current metrics to a historical CSV and generates a Plotly dashboard."""
    history_file = PROJECT_ROOT / "SIM_Results" / "Performance_History.csv"
    plot_file = PROJECT_ROOT / "SIM_Results" / "Performance_Dashboard.html"

    # 1. Extract the requested metrics from the current summary
    metrics_map = {row['Metric']: row['Value'] for _, row in summary_df.iterrows()}

    current_data = {
        "Timestamp": datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "Total Trades": metrics_map.get("Total Trades", 0),
        "Winning Trades": metrics_map.get("Winning Trades", 0),
        "Losing Trades": metrics_map.get("Losing Trades", 0),
        "Win Rate": metrics_map.get("Win Rate", 0.0),
        "Net Realized PnL": metrics_map.get("Net Realized PnL", 0.0),
        "Gross Profit": metrics_map.get("Gross Profit", 0.0),
        "Gross Loss": metrics_map.get("Gross Loss", 0.0),
        "Profit Factor": metrics_map.get("Profit Factor (Gross Profit/Gross Loss)", 0.0),
        "Expectancy per Trade": metrics_map.get("Expectancy per Trade", 0.0)
    }

    # 2. Append to CSV
    file_exists = history_file.exists()
    with open(history_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=current_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(current_data)

    # 3. Read the last 100 entries for plotting
    df_hist = pd.read_csv(history_file).tail(100)

    # 4. Generate Plotly Subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Net PnL & Expectancy ($)", "Win Rate (%) & Profit Factor", "Trade Distribution"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    x_vals = df_hist["Timestamp"]

    # Row 1: PnL (Bar) and Expectancy (Line)
    fig.add_trace(go.Bar(x=x_vals, y=df_hist["Net Realized PnL"], name="Net PnL", marker_color='rgba(38, 166, 154, 0.7)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df_hist["Expectancy per Trade"], name="Expectancy", line=dict(color='#ff9800', width=3)), row=1, col=1, secondary_y=True)

    # Row 2: Win Rate (Line) and Profit Factor (Line)
    fig.add_trace(go.Scatter(x=x_vals, y=df_hist["Win Rate"] * 100, name="Win Rate %", line=dict(color='#2196f3', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=df_hist["Profit Factor"], name="Profit Factor", line=dict(color='#e91e63', width=2, dash='dot')), row=2, col=1, secondary_y=True)

    # Row 3: Trade Counts (Stacked Area or Bars)
    fig.add_trace(go.Bar(x=x_vals, y=df_hist["Winning Trades"], name="Wins", marker_color='#4caf50'), row=3, col=1)
    fig.add_trace(go.Bar(x=x_vals, y=df_hist["Losing Trades"], name="Losses", marker_color='#f44336'), row=3, col=1)

    fig.update_layout(
        title="SID Equities v7.0: Rolling 100-Scan Performance Telemetry",
        template="plotly_dark",
        barmode='stack',
        height=900,
        hovermode="x unified"
    )

    fig.write_html(plot_file)
    print(f"  • Plotly Dashboard Logged  : {plot_file.name}")


def main():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("⚠️ Missing Alpaca API keys in .env file.")
        sys.exit(1)

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    # Load State Metadata
    state_file = PROJECT_ROOT / "data" / "alpaca_state.json"
    state_meta = {}
    hwm = float(account.portfolio_value)
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            try:
                state_data = json.load(f)
                state_meta = state_data.get("position_meta", {})
                hwm = state_data.get("hwm", hwm)
            except json.JSONDecodeError:
                pass

    # --- SECTION 1: LIVE ACCOUNT SNAPSHOT ---
    print("\n" + "=" * 100)
    print("🏦 ALPACA LIVE ACCOUNT SNAPSHOT")
    print("=" * 100)

    current_equity = float(account.portfolio_value)
    starting_equity = 100000.00
    all_time_pnl = current_equity - starting_equity
    pnl_pct = (all_time_pnl / starting_equity) * 100
    pnl_icon = "🟩" if all_time_pnl >= 0 else "🟥"
    non_margin_bp = getattr(account, 'non_marginable_buying_power', None) or getattr(account, 'cash', 0.0)
    drawdown = (hwm - current_equity) / hwm if hwm > 0 else 0
    is_defensive = drawdown >= 0.05

    print(f"  • Account Status       : {account.status}")
    print(f"  • Portfolio Value      : ${current_equity:,.2f}")
    print(f"  • {pnl_icon} All-Time P/L      : ${all_time_pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"  • Cash Balance         : ${float(account.cash):,.2f}")
    print(f"  • Non-Margin Buying Pwr: ${float(non_margin_bp):,.2f}")
    print(f"  • Drawdown Governor    : {drawdown * 100:.2f}% | Mode: {'🛡️ DEFENSIVE' if is_defensive else '🟢 NORMAL'}")

    # --- SECTION 2: OPEN POSITIONS & LIVE RSI ---
    export_open_positions = []
    total_unrealized_pnl = 0.0

    print("\n" + "=" * 100)
    print(f"📦 OPEN POSITIONS ({len(positions)})")
    print("=" * 100)

    if not positions:
        print("  • No active positions.")
    else:
        symbols = [p.symbol for p in positions]
        now_ny = datetime.now(NY_TZ)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Hour,
            start=now_ny - timedelta(days=40), feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(request_params).df
        current_rsis = {}

        if not bars.empty:
            bars.index.names = ["Ticker", "Datetime"]
            bars = bars.tz_convert("America/New_York", level="Datetime")
            bars.rename(columns={"close": "Close"}, inplace=True)

            for t in symbols:
                if t in bars.index.get_level_values("Ticker"):
                    df = bars.xs(t, level="Ticker").copy().sort_index()
                    if len(df) > 15:
                        delta = df['Close'].diff()
                        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                        ema_up = up.ewm(com=13, adjust=False).mean()
                        ema_down = down.ewm(com=13, adjust=False).mean()
                        df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
                        current_rsis[t] = df['RSI'].iloc[-1]

        # REVISED HEADER: Stop_Loss moved after Current
        print(f"{'Ticker':<7} {'Entry_Time':<19} {'Qty':<5} {'Avg_Entry':<10} {'Current':<10} {'Stop_Loss':<10} {'Unreal_PnL':<11} {'Ret_%':<8} {'Cur_RSI':<8} {'Exit_Rule'}")
        print("-" * 115)

        for p in positions:
            t = p.symbol
            qty = float(p.qty)
            entry = float(p.avg_entry_price)
            curr = float(p.current_price)
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100.0
            total_unrealized_pnl += pnl

            meta = state_meta.get(t, {})
            entry_time = meta.get('entry_time', 'N/A')
            sl_price = meta.get('sl_price', 0.0)

            # FORMAT STOP LOSS COLOR
            sl_str = f"${sl_price:.2f}" if sl_price else "N/A"
            sl_padded = f"{sl_str:<10}"
            if curr < entry:
                sl_display = f"\033[91m{sl_padded}\033[0m" # Red text, padded properly
            else:
                sl_display = sl_padded

            exit_rule = meta.get('exit_rule', 'N/A')
            c_rsi = current_rsis.get(t)
            rsi_str = f"{c_rsi:.1f}" if c_rsi is not None else "N/A"

            # REVISED PRINT ROW: sl_display inserted after curr
            print(f"{t:<7} {entry_time:<19} {int(qty):<5} ${entry:<9.2f} ${curr:<9.2f} {sl_display}${pnl:<10.2f} {pnl_pct:>+6.2f}%  {rsi_str:<8} {exit_rule}")

            export_open_positions.append({
                "Ticker": t, "Entry Time": entry_time, "Quantity": int(qty),
                "Avg Entry": entry, "Current Price": curr,
                "Stop Loss": sl_price if sl_price else "",
                "Unrealized PnL": pnl, "Return %": pnl_pct,
                "Current RSI": c_rsi if c_rsi is not None else "",
                "Exit Rule": exit_rule
            })

        print("-" * 115)
        print(f"TOTAL UNREALIZED PnL: ${total_unrealized_pnl:,.2f}")

    # --- SECTION 3: HISTORICAL PERFORMANCE METRICS ---
    ledger_file = PROJECT_ROOT / "SIM_Results" / "Alpaca_Live_Ledger.xlsx"
    summary_df, status = generate_performance_summary(ledger_file)

    print("\n" + "=" * 100)
    print("📈 HISTORICAL PERFORMANCE SUMMARY")
    print("=" * 100)

    if summary_df is not None:
        log_and_plot_performance_history(summary_df)
        print("-" * 100)

        for _, row in summary_df.iterrows():
            val = row['Value']
            # Basic formatting for terminal output
            if isinstance(val, float) and pd.notna(val):
                if "Rate" in row['Metric'] or "Return" in row['Metric']:
                    val_str = f"{val:.2%}" if "Rate" in row['Metric'] else f"{val:.2f}%"
                elif any(keyword in row['Metric'] for keyword in ["PnL", "Profit", "Loss", "Win", "Expectancy"]) and "Factor" not in row['Metric'] and "Trades" not in row['Metric']:
                    val_str = f"${val:,.2f}"
                else:
                    val_str = f"{val:,.2f}"
            else:
                val_str = str(val)
            print(f"  • {row['Metric']:<40}: {val_str}")
    else:
        print(f"  • {status}")

    print("=" * 100)

    # --- SECTION 4: MULTI-TAB EXCEL EXPORT ---
    date_str = datetime.now().strftime("%d%b%y").lower()
    output_filename = PROJECT_ROOT / "SIM_Results" / f"Trade_Summary_{date_str}.xlsx"

    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        currency_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        num_fmt = workbook.add_format({'num_format': '0.00'})
        bold_fmt = workbook.add_format({'bold': True})

        # 1. Export Open Positions Sheet
        if export_open_positions:
            df_open = pd.DataFrame(export_open_positions)
            df_open.to_excel(writer, sheet_name="Open_Positions", index=False)
            worksheet_open = writer.sheets["Open_Positions"]

            # Apply formatting to Open Positions
            worksheet_open.set_column('A:B', 15)
            worksheet_open.set_column('D:F', 12, currency_fmt)
            worksheet_open.set_column('G:G', 10, pct_fmt)
            worksheet_open.set_column('I:I', 12, currency_fmt)

        # 2. Export Performance Summary Sheet
        if summary_df is not None:
            summary_df.to_excel(writer, sheet_name="Performance_Summary", index=False)
            worksheet_summary = writer.sheets["Performance_Summary"]

            worksheet_summary.set_column('A:A', 45, bold_fmt)
            worksheet_summary.set_column('B:B', 20)

            for row_num, metric in enumerate(summary_df['Metric']):
                val = summary_df.iloc[row_num]['Value']
                cell = f'B{row_num + 2}'

                if pd.isna(val):
                    worksheet_summary.write(cell, "N/A")
                    continue

                if any(keyword in metric for keyword in ["PnL", "Profit", "Loss", "Win", "Expectancy"]) and "Rate" not in metric and "Trades" not in metric and "Factor" not in metric:
                    if isinstance(val, (int, float)):
                        worksheet_summary.write(cell, val, currency_fmt)
                elif "Rate" in metric:
                    worksheet_summary.write(cell, val, pct_fmt)
                elif "Factor" in metric or "Time" in metric or "Return" in metric:
                    if isinstance(val, (int, float)):
                        worksheet_summary.write(cell, val, num_fmt)
                else:
                    worksheet_summary.write(cell, val)

    print(f"\n[✓] Telemetry successfully exported to: {output_filename.name}")

if __name__ == "__main__":
    main()