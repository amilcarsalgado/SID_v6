"""
===============================================================================
SCRIPT: IMMEDIATE TAKE-PROFIT SWEEP
LOCATION: scripts/force_exits.py
DESCRIPTION: Instantly evaluates open positions and sells if RSI targets are met,
             bypassing the hourly daemon schedule.
===============================================================================
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import zoneinfo
import pandas as pd
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("APCA_API_KEY_ID") and os.getenv("ALPACA_API_KEY"):
    os.environ["APCA_API_KEY_ID"] = os.getenv("ALPACA_API_KEY")
if not os.getenv("APCA_API_SECRET_KEY") and os.getenv("ALPACA_SECRET_KEY"):
    os.environ["APCA_API_SECRET_KEY"] = os.getenv("ALPACA_SECRET_KEY")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest, GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
STATE_FILE = PROJECT_ROOT / "data" / "alpaca_state.json"

def append_to_ledger(trade_record: dict):
    ledger_dir = PROJECT_ROOT / "SIM_Results"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "Alpaca_Live_Ledger.xlsx"

    df_new = pd.DataFrame([trade_record])

    if ledger_path.exists():
        try:
            df_existing = pd.read_excel(ledger_path, sheet_name="Closed_Trades")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new

        try:
            with pd.ExcelWriter(ledger_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_combined.to_excel(writer, sheet_name="Closed_Trades", index=False)
            print(f"   📝 Ledger Updated: Logged terminated trade for {trade_record['Ticker']} to SIM_Results/Alpaca_Live_Ledger.xlsx")
        except Exception as e:
            print(f"   ❌ Error writing to Excel ledger: {e}")
    else:
        df_combined = df_new
        try:
            with pd.ExcelWriter(ledger_path, engine='openpyxl') as writer:
                df_combined.to_excel(writer, sheet_name="Closed_Trades", index=False)
            print(f"   📝 Ledger Created & Updated: Logged trade for {trade_record['Ticker']} to SIM_Results/Alpaca_Live_Ledger.xlsx")
        except Exception as e:
            print(f"   ❌ Error writing to Excel ledger: {e}")

def run_instant_exit_sweep():
    print("=" * 80)
    print("🚨 INITIATING INSTANT TAKE-PROFIT SWEEP...")
    print("=" * 80)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        print("⚠️ Could not load state file. Exiting to prevent data corruption.")
        return

    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    if not positions:
        print("📦 No open positions to evaluate. Exiting.")
        return

    symbols = list(positions.keys())
    now_ny = datetime.now(NY_TZ)
    start_time = now_ny - timedelta(days=60)

    print(f"📊 Fetching live bar data for {len(symbols)} positions...")
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Hour, start=start_time, feed=DataFeed.IEX)
    bars = data_client.get_stock_bars(req).df

    if bars.empty:
        print("⚠️ No bar data returned. Exiting.")
        return

    bars.index.names = ["Ticker", "Datetime"]
    bars = bars.tz_convert("America/New_York", level="Datetime")
    bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

    for t, pos in positions.items():
        if t not in bars.index.get_level_values("Ticker"): continue

        df = bars.xs(t, level="Ticker").copy().sort_index()
        if len(df) < 14: continue

        # Calculate instant RSI
        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))

        current_rsi = df['RSI'].iloc[-1]
        meta = state.get('position_meta', {}).get(t, {})
        target_rule = meta.get('exit_rule', 'RSI_50')
        target_rsi = 60 if target_rule == 'RSI_60_WHALE' else 50

        print(f"🔍 Evaluated {t}: Live RSI = {current_rsi:.1f} (Target: {target_rsi})")

        if current_rsi >= target_rsi:
            print(f"   🚀 [TAKE PROFIT MET] Selling {t} instantly!")

            # Cancel stops
            open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[t]))
            if open_orders:
                for order in open_orders:
                    trading_client.cancel_order_by_id(order.id)
                time.sleep(3)

            # Close position
            avg_entry = float(pos.avg_entry_price)
            qty = float(pos.qty)
            est_exit_price = float(pos.current_price)

            trading_client.close_position(t, close_options=ClosePositionRequest(percentage="100"))

            # Ledger update
            trade_record = {
                'Ticker': t,
                'Entry_Time': meta.get('entry_time', 'N/A'),
                'Exit_Time': now_ny.strftime('%Y-%m-%d %H:%M:%S'),
                'Shares': qty,
                'Avg_Entry': avg_entry,
                'SL_Price': meta.get('sl_price', 0.0),
                'Exit_Price': est_exit_price,
                'Realized_PnL': (est_exit_price - avg_entry) * qty,
                'Return_%': ((est_exit_price - avg_entry) / avg_entry) * 100,
                'Entry_RSI': meta.get('Entry_RSI', 'N/A'),
                'Entry_MACD': meta.get('Entry_MACD', 'N/A'),
                'Chart_Link': f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={t}"
            }
            append_to_ledger(trade_record)

            if t in state['position_meta']:
                del state['position_meta'][t]

    # Save cleanup
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

    print("✅ Sweep complete.")

if __name__ == "__main__":
    run_instant_exit_sweep()