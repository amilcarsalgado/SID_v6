"""
===============================================================================
SCRIPT: READ-ONLY ACCOUNT SNAPSHOT & LIVE RSI MONITOR
LOCATION: scripts/check_account.py
===============================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import zoneinfo
import pandas as pd
import numpy as np
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

    # Load state for metadata (Stop Loss, Exit Rule)
    state_file = PROJECT_ROOT / "data" / "alpaca_state.json"
    state_meta = {}
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            try:
                state_data = json.load(f)
                state_meta = state_data.get("position_meta", {})
                hwm = state_data.get("hwm", float(account.portfolio_value))
            except json.JSONDecodeError:
                hwm = float(account.portfolio_value)
    else:
        hwm = float(account.portfolio_value)

    print("\n" + "=" * 100)
    print("🏦 ALPACA LIVE ACCOUNT SNAPSHOT (READ-ONLY WITH LIVE RSI)")
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
    print(f"  • Buying Power         : ${float(account.buying_power):,.2f}")
    print(f"  • Non-Margin Buying Pwr: ${float(non_margin_bp):,.2f}")
    print(f"  • Drawdown Governor    : {drawdown * 100:.2f}% | Mode: {'🛡️ DEFENSIVE' if is_defensive else '🟢 NORMAL'}")
    print("=" * 100)

    if not positions:
        print("📦 Open Positions: 0")
        return

    # --- FETCH HISTORICAL DATA FOR CURRENT RSI ---
    symbols = [p.symbol for p in positions]
    now_ny = datetime.now(NY_TZ)
    start_time = now_ny - timedelta(days=40)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Hour,
        start=start_time,
        feed=DataFeed.IEX
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

    # --- PRINT TABLE ---
    print(f"📦 Open Positions ({len(positions)}):")
    print(f"{'Ticker':<7} {'Entry_Time':<19} {'Qty':<5} {'Avg_Entry':<10} {'Current':<10} {'Unreal_PnL':<11} {'Ret_%':<8} {'Cur_RSI':<8} {'Stop_Loss':<10} {'Exit_Rule':<13} {'Chart_Link'}")
    print("-" * 145)

    total_unrealized_pnl = 0.0

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
        sl_str = f"${sl_price:.2f}" if sl_price else "N/A"
        exit_rule = meta.get('exit_rule', 'N/A')

        c_rsi = current_rsis.get(t)
        rsi_str = f"{c_rsi:.1f}" if c_rsi is not None else "N/A"

        tv_url = f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={t}"

        print(f"{t:<7} {entry_time:<19} {int(qty):<5} ${entry:<9.2f} ${curr:<9.2f} ${pnl:<10.2f} {pnl_pct:>+6.2f}%  {rsi_str:<8} {sl_str:<10} {exit_rule:<13} {tv_url}")

    print("-" * 145)
    print(f"TOTAL UNREALIZED PnL: ${total_unrealized_pnl:,.2f}")

if __name__ == "__main__":
    main()