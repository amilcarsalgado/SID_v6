"""
===============================================================================
SCRIPT: BACKFILL MANUAL TRADES TO LEDGER (WITH SUMMARY TAB)
LOCATION: scripts/backfill_ledger.py
===============================================================================
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

def backfill_closed_trades():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

    trading_client = TradingClient(api_key, secret_key, paper=True)

    orders_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50)
    closed_orders = trading_client.get_orders(orders_req)

    target_symbols = {"BITO", "IBIT", "MSFT", "MSTR"}
    filled_trades = {}

    for order in closed_orders:
        if order.symbol in target_symbols and order.filled_at:
            if order.symbol not in filled_trades:
                filled_trades[order.symbol] = []
            filled_trades[order.symbol].append({
                'side': order.side,
                'qty': float(order.filled_qty) if order.filled_qty else 0,
                'price': float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                'time': order.filled_at
            })

    records = []
    for symbol in target_symbols:
        if symbol in filled_trades:
            fills = sorted(filled_trades[symbol], key=lambda x: x['time'])
            buys = [f for f in fills if f['side'] == 'buy']
            sells = [f for f in fills if f['side'] == 'sell']

            if buys and sells:
                entry = buys[0]
                exit_leg = sells[-1]
                pnl = (exit_leg['price'] - entry['price']) * entry['qty']
                ret_pct = ((exit_leg['price'] - entry['price']) / entry['price']) * 100

                records.append({
                    'Ticker': symbol,
                    'Entry_Time': str(entry['time']),
                    'Exit_Time': str(exit_leg['time']),
                    'Shares': entry['qty'],
                    'Avg_Entry': entry['price'],
                    'SL_Price': 0.0,
                    'Exit_Price': exit_leg['price'],
                    'Realized_PnL': pnl,
                    'Return_%': ret_pct,
                    'Entry_RSI': 'Manual',
                    'Entry_MACD': 'Manual',
                    'Chart_Link': f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={symbol}"
                })

    if not records:
        print("⚠️ No matching closed orders found in recent history.")
        return

    df_closed = pd.DataFrame(records)

    # Calculate Summary Metrics
    total_trades = len(df_closed)
    winning_trades = len(df_closed[df_closed['Realized_PnL'] > 0])
    losing_trades = len(df_closed[df_closed['Realized_PnL'] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
    total_realized_pnl = df_closed['Realized_PnL'].sum()

    gross_profits = df_closed[df_closed['Realized_PnL'] > 0]['Realized_PnL'].sum()
    gross_losses = abs(df_closed[df_closed['Realized_PnL'] < 0]['Realized_PnL'].sum())
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else float('inf')

    summary_data = [{
        'Total_Closed_Trades': total_trades,
        'Winning_Trades': winning_trades,
        'Losing_Trades': losing_trades,
        'Win_Rate_%': round(win_rate, 2),
        'Total_Realized_PnL': round(total_realized_pnl, 2),
        'Profit_Factor': round(profit_factor, 2)
    }]
    df_summary = pd.DataFrame(summary_data)

    ledger_dir = PROJECT_ROOT / "SIM_Results"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "Alpaca_Live_Ledger.xlsx"

    with pd.ExcelWriter(ledger_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_closed.to_excel(writer, sheet_name="Closed_Trades", index=False)

    print(f"✅ Successfully backfilled {len(records)} trades with Summary tab into {ledger_path}")

if __name__ == "__main__":
    backfill_closed_trades()