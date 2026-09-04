"""
===============================================================================
SCRIPT: LIGHTWEIGHT ALPACA RECONCILIATION & LEDGER EXPORTER
LOCATION: scripts/export_ledger.py
===============================================================================
"""

import os
import sys
import json
from collections import deque
from datetime import datetime
from pathlib import Path
import zoneinfo
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderSide
except ImportError:
    print("⚠️ Alpaca SDK not found. Install with: pip install alpaca-py")
    sys.exit(1)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def build_closed_trade_ledger():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ Missing API credentials in .env")
        return

    trading_client = TradingClient(api_key, secret_key, paper=True)

    print("\n" + "=" * 80)
    print("📑 ALPACA RECONCILIATION & LEDGER EXPORTER")
    print("=" * 80)
    print("🔍 Fetching broker order history from Alpaca...")

    # Fetch closed orders directly from broker (sorted chronologically)
    req = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=500,
        direction="asc"
    )

    try:
        orders = trading_client.get_orders(req)
    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        return

    if not orders:
        print("📦 No closed orders found on this account.")
        return

    # Load stop-loss metadata from state if available
    state_file = PROJECT_ROOT / "data" / "alpaca_state.json"
    state_meta = {}
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state_meta = json.load(f).get("position_meta", {})
        except Exception:
            pass

    # FIFO reconstruction of buy/sell pairs
    inventory = {}  # symbol -> deque of {'qty', 'price', 'time', 'sl'}
    closed_trades = []

    for o in orders:
        if o.filled_qty is None or float(o.filled_qty) == 0:
            continue

        sym = o.symbol
        qty = float(o.filled_qty)
        avg_price = float(o.filled_avg_price)
        filled_at = o.filled_at.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')

        if o.side == OrderSide.BUY:
            if sym not in inventory:
                inventory[sym] = deque()

            # Lookup stop-loss if known in state metadata
            sl_price = state_meta.get(sym, {}).get('sl_price', 0.0)
            inventory[sym].append({
                'qty': qty,
                'price': avg_price,
                'time': filled_at,
                'sl_price': sl_price
            })

        elif o.side == OrderSide.SELL:
            remaining_to_close = qty

            while remaining_to_close > 0 and sym in inventory and len(inventory[sym]) > 0:
                earliest_buy = inventory[sym][0]
                matched_qty = min(remaining_to_close, earliest_buy['qty'])

                pnl = (avg_price - earliest_buy['price']) * matched_qty
                ret_pct = ((avg_price - earliest_buy['price']) / earliest_buy['price']) * 100.0

                closed_trades.append({
                    'Ticker': sym,
                    'Entry_Time': earliest_buy['time'],
                    'Exit_Time': filled_at,
                    'Shares': int(matched_qty),
                    'Avg_Entry': round(earliest_buy['price'], 2),
                    'SL_Price': round(earliest_buy['sl_price'], 2),
                    'Exit_Price': round(avg_price, 2),
                    'Realized_PnL': round(pnl, 2),
                    'Return_%': round(ret_pct, 2),
                    'Chart_Link': f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={sym}"
                })

                earliest_buy['qty'] -= matched_qty
                remaining_to_close -= matched_qty

                if earliest_buy['qty'] <= 0:
                    inventory[sym].popleft()

    if not closed_trades:
        print("📦 No matched round-trip closed trades identified.")
        return

    df_trades = pd.DataFrame(closed_trades)

    # Summary metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['Realized_PnL'] > 0])
    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    total_realized_pnl = df_trades['Realized_PnL'].sum()

    summary_data = [
        {"Metric": "Total Closed Round-Trips", "Value": total_trades},
        {"Metric": "Winning Trades", "Value": winning_trades},
        {"Metric": "Losing Trades", "Value": total_trades - winning_trades},
        {"Metric": "Win Rate", "Value": f"{win_rate:.1f}%"},
        {"Metric": "Total Realized P&L", "Value": f"${total_realized_pnl:,.2f}"}
    ]
    df_summary = pd.DataFrame(summary_data)

    # Export to Excel
    export_dir = PROJECT_ROOT / "SIM_Results"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_file = export_dir / "Broker_Reconciliation.xlsx"

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_trades.to_excel(writer, sheet_name="Closed_Trades", index=False)

    print(f"✅ Successfully exported {len(df_trades)} closed trades to:")
    print(f"   📁 {out_file}\n")

    # CLI Output table
    print(f"{'Ticker':<7} {'Entry_Time':<19} {'Exit_Time':<19} {'Shares':<6} {'Entry':<9} {'Exit':<9} {'PnL':<10} {'Return_%'}")
    print("-" * 92)
    for _, row in df_trades.iterrows():
        pnl_str = f"${row['Realized_PnL']:,.2f}"
        ret_str = f"{row['Return_%']:+.2f}%"
        print(f"{row['Ticker']:<7} {row['Entry_Time']:<19} {row['Exit_Time']:<19} {row['Shares']:<6} ${row['Avg_Entry']:<8.2f} ${row['Exit_Price']:<8.2f} {pnl_str:<10} {ret_str}")
    print("-" * 92)
    print(f"TOTAL REALIZED P&L: ${total_realized_pnl:,.2f}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    build_closed_trade_ledger()