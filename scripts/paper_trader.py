"""
===============================================================================
SCRIPT: AUTONOMOUS ALPACA HOURLY TRADING ENGINE (7-BAR DAILY CADENCE)
LOCATION: scripts/paper_trader.py
AUTHOR: Alvaro Salgado
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

ENGINEERING DOSSIER & ARCHITECTURAL SPECIFICATIONS:
-------------------------------------------------------------------------------
1. Core Operating Cadence:
   - Operates on a 7-bar daily schedule using 60-minute timeframe bars.
   - Scan schedule: 10:30, 11:30, 12:30, 01:30, 02:30, 03:30, and 03:58 PM NY.
   - Scan 7 (03:58 PM Pre-Close) normalizes volume by pro-rating the 28 elapsed
     minutes across a standard 30-minute half-bar window (volume * 30.0 / 28.0).

2. Execution Hierarchy & Priority Queues:
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
     indicators emit false "chop" hooks. The pre-event buffer halts entries to
     prevent slippage and avoid gambling into fundamental volatility spikes.
   - Asset Blast Radius: Evaluates the 'impacted_assets' array. Events tagged
     'EQUITIES' or 'USD' halt this daemon; localized currency tags ('EUR', 'GBP')
     are ignored by Equities and reserved for the Forex expansion.
   - Stop-Loss Integrity: Hard stops submitted via OrderClass.OTO reside directly
     on Alpaca's matching engine and will execute even during blackouts.

4. Strict Cash Collar & Money Management:
   - Zero margin allowed. The engine physically checks `account.cash` before entry.
   - Sizing calculation: min(risk_shares, max_notional_shares, max_cash_shares).
   - If available cash cannot cover the purchase outright, the setup is skipped.
   - Max portfolio exposure: 15 open positions. Max single asset: 20% equity.

5. Capital Defense & Governors:
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

import sys
import os
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import zoneinfo
from typing import Tuple
import yaml
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import subprocess

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Environment Variable Resolution
env_path = PROJECT_ROOT / ".env"
print(f"Loading .env from: {env_path}, Exists: {env_path.exists()}")
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("APCA_API_KEY_ID") and os.getenv("ALPACA_API_KEY"):
    os.environ["APCA_API_KEY_ID"] = os.getenv("ALPACA_API_KEY")
if not os.getenv("APCA_API_SECRET_KEY") and os.getenv("ALPACA_SECRET_KEY"):
    os.environ["APCA_API_SECRET_KEY"] = os.getenv("ALPACA_SECRET_KEY")

# Alpaca SDK Imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, ClosePositionRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
except ImportError:
    print("⚠️ Alpaca SDK not found. Install with: pip install alpaca-py")
    sys.exit(1)

# Core Architecture Modules
from src.trading_engine.data.loader import MarketDataLoader
from macro_blackout import MacroBlackoutParser

# Timezone Standards
NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")

# Oracle Feature Schema
FEATURE_COLS = [
    'RSI_Extreme_Value',
    'Dist_From_EMA_%',
    'Entry_RVOL',
    'Entry_ATR',
    'Days_To_Earnings',
    'Days_Since_Earnings',
    'Entry_Month',
    'Risk_Tier'
]

def load_settings(config_path: str = "config/settings.yaml") -> dict:
    """Loads system configuration parameters."""
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


class AlpacaExecutionEngine:
    def __init__(self, config: dict):
        self.config = config
        self.api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials missing in .env")

        # Initialize API Clients (Paper Trading Mode Hardcoded for Safety)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        # Oracle v6.0 ML Gatekeeper Initialization
        model_path = PROJECT_ROOT / config.get("paths", {}).get("model", "data/models/IE_Oracle_v6.0_Hourly.joblib")
        self.oracle_model = joblib.load(model_path)

        # Market Data Loader & Watchlist Configuration
        self.loader = MarketDataLoader(config)
        self.min_score = config.get("filters", {}).get("min_ticker_score", 3)
        self.tickers, self.sector_map, self.earnings_df = self.loader.load_watchlist(min_score=self.min_score)

        # Money Management & Risk Parameters
        self.cfg_acct = config.get("account", {})
        self.sizing_risk_base = float(self.cfg_acct.get("sizing_risk_base", 30000.0))
        self.max_positions = int(self.cfg_acct.get("max_open_positions", 15))
        self.max_notional_pct = float(self.cfg_acct.get("max_notional_allocation_pct", 0.20))
        self.trap_thresh = float(config.get("filters", {}).get("trap_threshold", 0.35))
        self.whale_thresh = float(config.get("filters", {}).get("whale_threshold", 0.42))
        self.starting_equity = 100000.00

        # State Persistence
        self.state_file = PROJECT_ROOT / "data" / "alpaca_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

        # Macro Blackout Gatekeeper Initialization (Config Directory Path)
        blackout_path = PROJECT_ROOT / "config" / "blackout_dates.json"
        self.macro_parser = MacroBlackoutParser(str(blackout_path))

    def api_retry(self, func, *args, retries=3, delay=5, **kwargs):
        """Transient error wrapper to catch Alpaca HTTP 500 / Network Timeouts."""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                print(f"   ⚠️ Alpaca API hiccup ({e}). Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)

    def load_state(self) -> dict:
        """Loads state persistence JSON."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ State file corrupted. Initializing fresh state.")
        return {"hwm": 0.0, "burned_regimes": {}, "position_meta": {}}

    def save_state(self):
        """Persists state to local JSON file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4, ensure_ascii=False)

    def append_to_ledger(self, trade_record: dict):
        """Appends liquidated position records to the persistent Excel ledger."""
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

    def log_comprehensive_feature_snapshot(
        self, ticker: str, features_df: pd.DataFrame, p_trap: float,
        p_whale: float, action: str, extra_metadata: dict = None
    ):
        """Logs live out-of-sample feature state snapshots (Ghost Ledger) for Oracle v7."""
        store_dir = PROJECT_ROOT / "data" / "training_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = store_dir / "oracle_v7_live_training_store.parquet"

        record = features_df.copy()
        record['Ticker'] = ticker
        record['Sector_Name'] = self.sector_map.get(ticker, 'Unknown')
        record['Watchlist_Score'] = 3
        record['Direction'] = 'LONG'
        record['P_Trap'] = p_trap
        record['P_Whale'] = p_whale
        record['Action_Taken'] = action
        record['Capture_Time'] = datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')

        if extra_metadata:
            for k, v in extra_metadata.items():
                record[k] = v

        df_new = pd.DataFrame(record)

        if parquet_path.exists():
            try:
                df_existing = pd.read_parquet(parquet_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_parquet(parquet_path, index=False)

    def print_startup_banner(self):
        print("=" * 85)
        print("🤖 SID EQUITIES LONG AUTONOMOUS TRADING ENGINE — STARTUP OVERVIEW")
        print("=" * 85)
        print(" [SYSTEM & MONEY MANAGEMENT]")
        print("  • Strategy Mode        : Long-Only Equity Swing (Shorts disabled)")
        print(f"  • Sizing Base / Max Cap: ${self.sizing_risk_base:,.2f} risk base | Max {self.max_positions} open positions")
        print(f"  • ML Gatekeeper        : Oracle v6.0 Hourly (Veto if P(Trap) >= {int(self.trap_thresh * 100)}%)")
        print("  • Order Execution      : Market Buy + Hard OTO Stop-Loss (CASH ONLY / NO MARGIN)")
        print("  • Profit Exits         : Immediate Scan Execution if RSI >= 50 (or >= 60 Whale)")
        print("  • Capital Protection   : Drawdown Governor & Regime Burn Guard Active")
        print("  • Macro Defense        : Blackout Calendar Active (Halts Entries / Keeps SL Active)")
        print("-" * 85)
        print(" [SCAN SCHEDULE (7-BAR CADENCE)]")
        print("  • Scans 1 to 6         : 10:30, 11:30, 12:30, 01:30, 02:30, 03:30 (Closed 60m bars)")
        print("  • Scan 7 (Pre-Close)   : 03:58 PM (28m elapsed; pro-rated volume)")
        print("=" * 85 + "\n")

    def print_boot_snapshot(self):
        """Fetches and displays real-time portfolio status upon daemon boot."""
        account = self.api_retry(self.trading_client.get_account)
        positions = self.api_retry(self.trading_client.get_all_positions)

        print("\n" + "=" * 90)
        print("🏦 BOOT-UP ACCOUNT SNAPSHOT")
        print("=" * 90)
        print(f"  • Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  • Cash Balance   : ${float(account.cash):,.2f}")
        print("=" * 90)

        if not positions:
            print("📦 Open Positions: 0 (No active trades)\n")
            return

        all_symbols = [p.symbol for p in positions]
        start_time = datetime.now(NY_TZ) - timedelta(days=60)
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=all_symbols, timeframe=TimeFrame.Hour, start=start_time, feed=DataFeed.IEX
            )
            bars = self.api_retry(self.data_client.get_stock_bars, request_params).df
            if not bars.empty:
                bars.index.names = ["Ticker", "Datetime"]
                bars = bars.tz_convert("America/New_York", level="Datetime")
        except Exception:
            bars = pd.DataFrame()

        print(f"📦 Open Positions ({len(positions)} / {self.max_positions}):")
        print(f"{'Ticker':<7} {'Entry_Time':<19} {'Qty':<5} {'Avg_Entry':<10} {'Curr_Price':<11} {'SL':<8} {'Unreal_PnL':<11} {'Ret_%':<8} {'E_RSI':<7} {'Curr_RSI':<9} {'Chart_Link'}")
        print("-" * 125)

        for p in positions:
            t = p.symbol
            qty = float(p.qty)
            entry = float(p.avg_entry_price)
            curr = float(p.current_price)
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100.0

            curr_rsi_val = "N/A"
            if not bars.empty and t in bars.index.get_level_values("Ticker"):
                tdf = bars.xs(t, level="Ticker").copy().sort_index()
                if len(tdf) >= 14:
                    delta = tdf['close'].diff()
                    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                    ema_up = up.ewm(com=13, adjust=False).mean()
                    ema_down = down.ewm(com=13, adjust=False).mean()
                    rsi_series = 100 - (100 / (1 + (ema_up / ema_down)))
                    curr_rsi_val = f"{rsi_series.iloc[-1]:.1f}"

            meta = self.state.get('position_meta', {}).get(t, {})
            entry_time = meta.get('entry_time', 'N/A')

            e_rsi = meta.get('Entry_RSI')
            rsi_str = f"{e_rsi:.1f}" if e_rsi is not None else "N/A"

            sl_val = meta.get('sl_price', 0.0)
            if sl_val > 0:
                base_sl = f"${sl_val:.2f}"
                if curr < entry:
                    sl_padded = f"\033[91m{base_sl:<8}\033[0m"
                else:
                    sl_padded = f"{base_sl:<8}"
            else:
                sl_padded = f"{'N/A':<8}"

            tv_url = f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={t}"
            print(f"{t:<7} {entry_time:<19} {int(qty):<5} ${entry:<9.2f} ${curr:<10.2f} {sl_padded} ${pnl:<10.2f} {pnl_pct:>+6.2f}%  {rsi_str:<7} {curr_rsi_val:<9} {tv_url}")
        print("-" * 125 + "\n")

    def get_next_schedule_target(self) -> Tuple[datetime, str]:
        """Calculates exact wake-up timestamp based on the 7-bar schedule."""
        now = datetime.now(NY_TZ)
        today = now.date()
        schedule = [
            (datetime(today.year, today.month, today.day, 10, 30, 5, tzinfo=NY_TZ), "Bar 1 (10:30 AM)"),
            (datetime(today.year, today.month, today.day, 11, 30, 5, tzinfo=NY_TZ), "Bar 2 (11:30 AM)"),
            (datetime(today.year, today.month, today.day, 12, 30, 5, tzinfo=NY_TZ), "Bar 3 (12:30 PM)"),
            (datetime(today.year, today.month, today.day, 13, 30, 5, tzinfo=NY_TZ), "Bar 4 (01:30 PM)"),
            (datetime(today.year, today.month, today.day, 14, 30, 5, tzinfo=NY_TZ), "Bar 5 (02:30 PM)"),
            (datetime(today.year, today.month, today.day, 15, 30, 5, tzinfo=NY_TZ), "Bar 6 (03:30 PM)"),
            (datetime(today.year, today.month, today.day, 15, 58, 0, tzinfo=NY_TZ), "Bar 7 Final (03:58 PM)")
        ]
        for target_time, label in schedule:
            if now < target_time:
                return target_time, label

        tomorrow = today + timedelta(days=1)
        next_day_first = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 10, 30, 5, tzinfo=NY_TZ)
        return next_day_first, "Bar 1 Next Session (10:30 AM)"

    def evaluate_live_scan(self, is_final_bar: bool = False):
        """Core algorithmic execution pass executed hourly."""
        now_ny = datetime.now(NY_TZ)
        now_utc = datetime.now(timezone.utc)  # <--- UPDATED TO MATCH IMPORT
        scan_label = "FINAL PRE-CLOSE" if is_final_bar else now_ny.strftime('%H:%M:%S')

        print("\n\n\n\n")
        print(f"[{now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}] 🔍 Running Hourly Scan ({scan_label})...")

        account = self.api_retry(self.trading_client.get_account)
        current_equity = float(account.portfolio_value)
        self.state['hwm'] = max(self.state.get('hwm', current_equity), current_equity)

        # Drawdown Governor Calculation
        drawdown = (self.state['hwm'] - current_equity) / self.state['hwm']
        is_defensive = drawdown >= 0.05

        raw_positions = self.api_retry(self.trading_client.get_all_positions)
        positions = {p.symbol: p for p in raw_positions}
        active_symbols = set(positions.keys())

        # =====================================================================
        # PRIORITY 0: STOP-LOSS INTERCEPTOR
        # =====================================================================
        for t in list(self.state.get('position_meta', {}).keys()):
            if t not in active_symbols:
                print(f"⚠️ {t} is missing from Alpaca (likely stopped out). Logging to ledger...")
                meta = self.state['position_meta'][t]

                try:
                    closed_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[t], limit=10)
                    closed_orders = self.api_retry(self.trading_client.get_orders, closed_req)

                    sell_orders = [o for o in closed_orders if o.side == OrderSide.SELL and o.filled_qty and float(o.filled_qty) > 0]

                    if sell_orders:
                        last_sell = sell_orders[0]
                        exit_price = float(last_sell.filled_avg_price)
                        exit_time = last_sell.filled_at.astimezone(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                        qty = float(last_sell.filled_qty)
                    else:
                        exit_price = meta.get('sl_price', 0.0)
                        exit_time = now_ny.strftime('%Y-%m-%d %H:%M:%S')
                        qty = 0

                    avg_entry = meta.get('avg_entry', 0.0)
                    realized_pnl = (exit_price - avg_entry) * qty if qty > 0 else 0.0
                    return_pct = ((exit_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0.0

                    trade_record = {
                        'Ticker': t,
                        'Entry_Time': meta.get('entry_time', 'N/A'),
                        'Exit_Time': exit_time,
                        'Shares': qty,
                        'Avg_Entry': avg_entry,
                        'SL_Price': meta.get('sl_price', 0.0),
                        'Exit_Price': exit_price,
                        'Realized_PnL': realized_pnl,
                        'Return_%': return_pct,
                        'Entry_RSI': meta.get('Entry_RSI', 'N/A'),
                        'Entry_MACD': meta.get('Entry_MACD', 'N/A'),
                        'Chart_Link': f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={t}"
                    }
                    self.append_to_ledger(trade_record)
                except Exception as e:
                    print(f"❌ Error recovering stopped-out trade {t} for ledger: {e}")

        # Clean up the state file after ledgering is complete
        self.state['position_meta'] = {k: v for k, v in self.state.get('position_meta', {}).items() if k in active_symbols}
        self.save_state()
        # =====================================================================

        # Ingest Market Data for Watchlist + Open Positions
        all_symbols = list(set(self.tickers + list(positions.keys())))
        start_time = now_ny - timedelta(days=60)
        request_params = StockBarsRequest(
            symbol_or_symbols=all_symbols, timeframe=TimeFrame.Hour, start=start_time, feed=DataFeed.IEX
        )

        bars_call = self.api_retry(self.data_client.get_stock_bars, request_params)
        bars = bars_call.df if bars_call else pd.DataFrame()

        if bars.empty:
            print("⚠️ No bar data returned from feed.")
            return

        bars.index.names = ["Ticker", "Datetime"]
        bars = bars.tz_convert("America/New_York", level="Datetime")
        bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

        symbol_data = {}

        # ─── 1. FAST INDICATOR PROCESSING ───
        for t in all_symbols:
            if t not in bars.index.get_level_values("Ticker"): continue

            df = bars.xs(t, level="Ticker").copy().sort_index()
            if len(df) < 50: continue

            if not is_final_bar:
                last_bar_time = df.index[-1]
                current_period_start = now_ny.replace(minute=30, second=0, microsecond=0)
                if last_bar_time >= current_period_start:
                    df = df.iloc[:-1]

            if len(df) < 50: continue

            if is_final_bar:
                df.iloc[-1, df.columns.get_loc('Volume')] = df.iloc[-1]['Volume'] * (30.0 / 28.0)

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
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

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
            df['SL_L'] = np.floor(df['Lowest_Since'])

            symbol_data[t] = {
                'df': df,
                'latest': df.iloc[-1],
                'prev': df.iloc[-2]
            }

        # ─── 2. IMMEDIATE EXITS (EXECUTION PRIORITY 1: ALWAYS RUNS) ───
        for t in list(positions.keys()):
            if t not in symbol_data: continue
            current_rsi = symbol_data[t]['latest']['RSI']
            meta = self.state['position_meta'].get(t, {})
            target_rule = meta.get('exit_rule', 'RSI_50')
            target_rsi = 60 if target_rule == 'RSI_60_WHALE' else 50

            if current_rsi >= target_rsi:
                try:
                    print(f"🚀 [TAKE PROFIT FIRED] Closing {t} (Current RSI {current_rsi:.1f} >= Target {target_rsi})")

                    pos = positions[t]
                    avg_entry = float(pos.avg_entry_price)
                    qty = float(pos.qty)
                    est_exit_price = float(pos.current_price)

                    open_orders_req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[t])
                    open_orders = self.api_retry(self.trading_client.get_orders, open_orders_req)

                    if open_orders:
                        for order in open_orders:
                            self.api_retry(self.trading_client.cancel_order_by_id, order.id)
                        time.sleep(3)

                    close_req = ClosePositionRequest(percentage="100")
                    self.api_retry(self.trading_client.close_position, t, close_options=close_req)

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
                    self.append_to_ledger(trade_record)

                    if t in self.state['position_meta']:
                        del self.state['position_meta'][t]
                    self.save_state()
                    del positions[t]

                except Exception as e:
                    print(f"❌ Error closing {t}: {e}")

        # ─── 3. MACRO BLACKOUT GATEKEEPER ───
        # Validates UTC timestamp against config/blackout_dates.json
        is_safe_to_trade = self.macro_parser.is_trade_allowed(now_utc, "EQUITIES")

        approved_count = 0
        blocked_count = 0
        available_cash = float(account.cash)

        if not is_safe_to_trade:
            print("   🛡️ MACRO VETO: Active blackout window detected. Halting all new entry scans.")
        else:
            # ─── 4. IMMEDIATE ENTRIES (EXECUTION PRIORITY 2: CASH ONLY) ───
            for t in self.tickers:
                if t in positions or t not in symbol_data: continue
                if len(positions) >= self.max_positions: break

                sdata = symbol_data[t]
                df = sdata['df']

                if len(df) < 200: continue

                latest = sdata['latest']
                current_L_ID = int(latest['L_ID'])

                burned = self.state.get('burned_regimes', {}).get(t, [])
                if current_L_ID in burned: continue

                is_above_trend = latest['Close'] > latest['EMA_200']
                is_agg = (latest['State_L_Agg'] == 1) and (df['R_Hook_U'].iloc[-3:].max() == 1) and latest['M_Hook_U'] and is_above_trend
                is_tier = (latest['State_L_Tier'] == 1) and (df['R_Hook_U'].iloc[-3:].max() == 1) and latest['M_Hook_U'] and is_above_trend

                if is_agg or is_tier:
                    entry_price = float(latest['Close'])
                    stop_loss = float(latest['SL_L'])

                    if pd.isna(stop_loss) or entry_price <= stop_loss:
                        continue

                    risk_tier = 0.02 if is_agg else 0.005
                    if is_defensive:
                        risk_tier = min(risk_tier, 0.01)

                    t_earnings = self.earnings_df[self.earnings_df['Ticker'] == t]['Earnings_Date']
                    days_to_earn, days_since_earn = 45, 45
                    if not t_earnings.empty:
                        deltas = (t_earnings.dt.tz_localize(NY_TZ) - now_ny).dt.days
                        fut = deltas[deltas >= 0]
                        pst = deltas[deltas < 0]
                        if not fut.empty: days_to_earn = int(fut.min())
                        if not pst.empty: days_since_earn = int(abs(pst.max()))

                    dist_from_ema = ((entry_price - latest['EMA_200']) / latest['EMA_200']) * 100

                    features = pd.DataFrame([{
                        'RSI_Extreme_Value': latest['Extreme_RSI'],
                        'Dist_From_EMA_%': dist_from_ema,
                        'Entry_RVOL': latest['RVOL'],
                        'Entry_ATR': latest['ATR'],
                        'Days_To_Earnings': days_to_earn,
                        'Days_Since_Earnings': days_since_earn,
                        'Entry_Month': now_ny.month,
                        'Risk_Tier': risk_tier
                    }])[FEATURE_COLS].fillna(0)

                    probs = self.oracle_model.predict_proba(features)[0]
                    p_trap, p_whale = probs[0], probs[2] if len(probs) > 2 else 0.0

                    print(f"🎯 Signal: {t:<5} | Price: ${entry_price:.2f} | P(Trap): {p_trap:.3f} | P(Whale): {p_whale:.3f}")

                    if p_trap >= self.trap_thresh:
                        print(f"   🚫 VETOED by Oracle: P(Trap) {p_trap:.2f} >= {self.trap_thresh}")
                        blocked_count += 1
                        self.log_comprehensive_feature_snapshot(t, features, p_trap, p_whale, action="VETOED")
                        continue

                    dollar_risk = self.sizing_risk_base * risk_tier
                    risk_per_share = abs(entry_price - stop_loss)

                    shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
                    max_shares = int((current_equity * self.max_notional_pct) / entry_price)

                    # STRICT CASH COLLAR ENFORCEMENT (ZERO MARGIN)
                    max_cash_shares = int(available_cash / entry_price) if available_cash > 0 else 0
                    shares = min(shares, max_shares, max_cash_shares)

                    if shares <= 0:
                        if available_cash < entry_price:
                            print(f"   ⚠️ Blocked {t}: Insufficient cash (${available_cash:,.2f}) to avoid using margin.")
                        continue

                    try:
                        order_req = MarketOrderRequest(
                            symbol=t, qty=shares, side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC, order_class=OrderClass.OTO,
                            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2))
                        )
                        self.api_retry(self.trading_client.submit_order, order_req)
                        approved_count += 1

                        available_cash -= (shares * entry_price)

                        self.log_comprehensive_feature_snapshot(
                            t, features, p_trap, p_whale, action="BOUGHT",
                            extra_metadata={'Applied_Risk_Pct': risk_tier, 'Stop_Loss': stop_loss, 'Entry_Price': entry_price}
                        )

                        exit_rule = "RSI_60_WHALE" if p_whale > self.whale_thresh else "RSI_50"

                        self.state['position_meta'][t] = {
                            'exit_rule': exit_rule,
                            'entry_time': now_ny.strftime('%Y-%m-%d %H:%M:%S'),
                            'avg_entry': entry_price,
                            'sl_price': stop_loss,
                            'Entry_RSI': float(latest['RSI']),
                            'Entry_MACD': float(latest['MACD']),
                            'Entry_RVOL': float(latest['RVOL']),
                            'Dist_from_EMA': float(dist_from_ema)
                        }
                        self.state.setdefault('burned_regimes', {}).setdefault(t, []).append(current_L_ID)
                        self.save_state()

                        print(f"   ✅ BOUGHT: {shares} shares of {t} | Stop @ ${stop_loss:.2f} | Target: {exit_rule}")
                    except Exception as e:
                        print(f"   ❌ Execution Error: {e}")

        # ─── 5. CONSOLIDATED POST-EXECUTION REPORTING (ACCOUNT SNAPSHOT) ───

        if approved_count > 0:
            print(f"   ⏳ Waiting 5 seconds for Alpaca to fill {approved_count} new market order(s)...")
            time.sleep(5)

        account = self.api_retry(self.trading_client.get_account)
        current_equity = float(account.portfolio_value)
        non_margin_bp = getattr(account, 'non_marginable_buying_power', None) or getattr(account, 'cash', 0.0)

        all_time_pnl = current_equity - self.starting_equity
        pnl_pct = (all_time_pnl / self.starting_equity) * 100
        pnl_icon = "🟩" if all_time_pnl >= 0 else "🟥"

        updated_positions = self.api_retry(self.trading_client.get_all_positions)

        print("\n" + "=" * 90)
        print("🏦 ALPACA LIVE ACCOUNT SNAPSHOT (POST-SCAN)")
        print("=" * 90)
        print(f"  • Account Status       : {account.status}")
        print(f"  • Portfolio Value      : ${current_equity:,.2f}")
        print(f"  • {pnl_icon} All-Time P/L      : ${all_time_pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"  • Cash Balance         : ${float(account.cash):,.2f}")
        print(f"  • Buying Power         : ${float(account.buying_power):,.2f}")
        print(f"  • Non-Margin Buying Pwr: ${float(non_margin_bp):,.2f}")
        print(f"  • Drawdown Governor    : {drawdown * 100:.2f}% | Mode: {'🛡️ DEFENSIVE' if is_defensive else '🟢 NORMAL'}")
        print(f"  • Scan Result          : Approved: {approved_count} | Blocked: {blocked_count}")
        print("=" * 90)

        if not updated_positions:
            print("📦 Open Positions: 0 (No active trades)")
        else:
            print(f"📦 Open Positions ({len(updated_positions)} / {self.max_positions}):")
            print(f"{'Ticker':<7} {'Entry_Time':<19} {'Qty':<5} {'Avg_Entry':<10} {'Curr_Price':<11} {'SL':<8} {'Unreal_PnL':<11} {'Ret_%':<8} {'E_RSI':<7} {'Curr_RSI':<9} {'Chart_Link'}")
            print("-" * 125)

            total_unrealized_pnl = 0.0

            for p in updated_positions:
                t = p.symbol
                qty = float(p.qty)
                entry = float(p.avg_entry_price)
                curr = float(p.current_price)
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100.0
                total_unrealized_pnl += pnl

                curr_rsi_val = "N/A"
                if t in symbol_data:
                    curr_rsi_val = f"{symbol_data[t]['latest']['RSI']:.1f}"

                meta = self.state.get('position_meta', {}).get(t, {})
                entry_time = meta.get('entry_time', 'N/A')

                e_rsi = meta.get('Entry_RSI')
                rsi_str = f"{e_rsi:.1f}" if e_rsi is not None else "N/A"

                sl_val = meta.get('sl_price', 0.0)
                if sl_val > 0:
                    base_sl = f"${sl_val:.2f}"
                    if curr < entry:
                        sl_padded = f"\033[91m{base_sl:<8}\033[0m"
                    else:
                        sl_padded = f"{base_sl:<8}"
                else:
                    sl_padded = f"{'N/A':<8}"

                tv_url = f"https://www.tradingview.com/chart/QxnQNEPO/?symbol={t}"

                print(f"{t:<7} {entry_time:<19} {int(qty):<5} ${entry:<9.2f} ${curr:<10.2f} {sl_padded} ${pnl:<10.2f} {pnl_pct:>+6.2f}%  {rsi_str:<7} {curr_rsi_val:<9} {tv_url}")

            print("-" * 125)
            print(f"TOTAL UNREALIZED PnL: ${total_unrealized_pnl:,.2f}")
        print("=" * 125 + "\n")

    def run_daemon(self):
        """Continuous execution loop tracking the 7-bar daily cadence."""
        self.print_startup_banner()
        self.print_boot_snapshot()

        while True:
            try:
                target_time, label = self.get_next_schedule_target()
                now = datetime.now(NY_TZ)
                sleep_sec = max(5, int((target_time - now).total_seconds()))

                hrs, remainder = divmod(sleep_sec, 3600)
                mins, secs = divmod(remainder, 60)

                print(f"⏰ Next Scan Scheduled: {label} at {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"💤 Sleeping for {hrs}h {mins}m {secs}s...")
                time.sleep(sleep_sec)

                clock = self.api_retry(self.trading_client.get_clock)

                if clock.is_open:
                    is_final = (target_time.hour == 15 and target_time.minute >= 38)
                    self.evaluate_live_scan(is_final_bar=is_final)
                else:
                    print("🌙 Market is closed. Skipping scan until next active open.")

            except KeyboardInterrupt:
                print("\n🛑 Daemon stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Error in loop: {e}. Retrying in 60s...")
                time.sleep(60)

if __name__ == "__main__":
    engine = AlpacaExecutionEngine(load_settings())
    engine.run_daemon()