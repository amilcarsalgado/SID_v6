"""
===============================================================================
SCRIPT: HOURLY VECTORIZED BACKTEST ENGINE WITH ML INFERENCE
LOCATION: scripts/run_backtest.py
===============================================================================
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import yaml
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_engine.data.loader import MarketDataLoader

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
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


class HourlyPortfolio:
    """Simulates realistic execution with cash tracking, sizing, and the drawdown governor."""
    def __init__(self, config: dict, sector_map: dict):
        self.cfg_acct = config.get("account", {})
        self.initial_balance = float(self.cfg_acct.get("initial_cash", 40000.0))
        self.balance = self.initial_balance
        self.current_equity = self.initial_balance
        self.sizing_risk_base = float(self.cfg_acct.get("sizing_risk_base", 30000.0))
        self.max_open_positions = int(self.cfg_acct.get("max_open_positions", 15))
        self.max_notional_pct = float(self.cfg_acct.get("max_notional_allocation_pct", 0.20))
        self.commission = float(self.cfg_acct.get("commission", 1.00))
        self.defensive_risk_pct = float(self.cfg_acct.get("defensive_risk_pct", 0.01))
        self.drawdown_threshold = float(self.cfg_acct.get("drawdown_threshold", 0.05))

        self.sector_name_map = sector_map
        self.high_water_mark = self.initial_balance
        self.positions = {}
        self.master_log = []
        self.burned_regimes = {}

    def can_open_position(self, ticker: str, regime_id: int) -> bool:
        if len(self.positions) >= self.max_open_positions:
            return False
        if ticker in self.positions:
            return False
        if regime_id in self.burned_regimes.get(ticker, []):
            return False
        return True

    def execute_buy(self, dt, ticker: str, entry_price: float, stop_loss: float,
                    extreme_rsi: float, regime_id: int, entry_ema: float, risk_tier: float,
                    entry_rvol: float, entry_atr: float, target_rsi: int,
                    days_to_earn: int, days_since_earn: int,
                    p_trap: float, p_whale: float):
        if pd.isna(entry_price) or pd.isna(stop_loss) or entry_price <= stop_loss:
            return

        self.high_water_mark = max(self.high_water_mark, self.current_equity)
        current_drawdown = (self.high_water_mark - self.current_equity) / self.high_water_mark

        is_gov_active = current_drawdown >= self.drawdown_threshold
        active_risk_pct = self.defensive_risk_pct if is_gov_active else risk_tier

        dollar_risk = self.sizing_risk_base * active_risk_pct
        risk_per_share = abs(entry_price - stop_loss)
        max_shares_by_risk = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

        max_notional_dollars = self.current_equity * self.max_notional_pct
        max_shares_by_cap = int(max_notional_dollars / entry_price) if entry_price > 0 else 0

        shares = min(max_shares_by_risk, max_shares_by_cap)
        position_cost = shares * entry_price

        if shares > 0 and (position_cost + self.commission) <= self.balance:
            self.balance -= (position_cost + self.commission)
            self.positions[ticker] = {
                'entry_datetime': dt,
                'entry_date': dt.date(),
                'shares': shares,
                'entry_price': entry_price,
                'invested_capital': position_cost,
                'stop_loss': stop_loss,
                'extreme_rsi': extreme_rsi,
                'regime_id': regime_id,
                'applied_risk_pct': active_risk_pct,
                'entry_ema': entry_ema,
                'entry_rvol': entry_rvol,
                'entry_atr': entry_atr,
                'target_rsi': target_rsi,
                'days_to_earn': days_to_earn,
                'days_since_earn': days_since_earn,
                'p_trap': p_trap,
                'p_whale': p_whale
            }
            self.burned_regimes.setdefault(ticker, []).append(regime_id)

    def execute_sell(self, dt, ticker: str, exit_price: float, reason: str, exit_rsi: float):
        pos = self.positions.pop(ticker)
        gain_per_share = exit_price - pos['entry_price']
        total_profit = (gain_per_share * pos['shares']) - (self.commission * 2)

        cash_returned = pos['shares'] * exit_price
        self.balance += (cash_returned - self.commission)

        invested_at_exit = sum(p['invested_capital'] for p in self.positions.values())
        self.current_equity = self.balance + invested_at_exit
        self.high_water_mark = max(self.high_water_mark, self.current_equity)

        log_entry = {
            'Ticker': ticker,
            'Sector_Name': self.sector_name_map.get(ticker, 'Unmapped'),
            'Direction': 'LONG',
            'Applied_Risk_Pct': f"{pos['applied_risk_pct'] * 100:.1f}%",
            'RSI_Extreme_Value': round(pos['extreme_rsi'], 2),
            'Entry_Date': str(pos['entry_date']),
            'Entry_Time': pos['entry_datetime'].strftime("%H:%M:%S"),
            'Entry_Price': round(pos['entry_price'], 2),
            'Entry_EMA_200': round(pos['entry_ema'], 2),
            'Dist_From_EMA_%': round(((pos['entry_price'] - pos['entry_ema']) / pos['entry_ema']) * 100, 2),
            'Entry_RVOL': round(pos['entry_rvol'], 2),
            'Entry_ATR': round(pos['entry_atr'], 2),
            'Exit_Date': str(dt.date()),
            'Exit_Time': dt.strftime("%H:%M:%S"),
            'Hold_Duration_Hours': int((dt - pos['entry_datetime']).total_seconds() // 3600),
            'Exit_Price': round(exit_price, 2),
            'Exit_RSI': round(exit_rsi, 2),
            'Exit_Reason': reason,
            'Target_RSI_Assigned': pos['target_rsi'],
            'P_Trap': round(pos['p_trap'], 3),
            'P_Whale': round(pos['p_whale'], 3),
            'Shares': pos['shares'],
            'Net_PnL (After Fees)': round(total_profit, 2),
            'Account_Equity': round(self.current_equity, 2)
        }
        self.master_log.append(log_entry)

    def export_results(self, output_dir: Path, revision: str):
        if not self.master_log:
            print("⚠️ No trades executed during simulation.")
            return

        master_df = pd.DataFrame(self.master_log).sort_values(by=['Entry_Date', 'Ticker'])
        total_trades = len(master_df)
        wins = master_df[master_df['Net_PnL (After Fees)'] > 0]
        losses = master_df[master_df['Net_PnL (After Fees)'] <= 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = wins['Net_PnL (After Fees)'].sum()
        gross_loss = abs(losses['Net_PnL (After Fees)'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        capital_invested = sum(p['invested_capital'] for p in self.positions.values())
        total_return = ((self.current_equity - self.initial_balance) / self.initial_balance) * 100
        unique_dates = master_df['Entry_Date'].unique()

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
                revision,
                len(unique_dates),
                round(len(unique_dates) / 21, 1),
                f"${self.initial_balance:,.2f}",
                f"${self.current_equity:,.2f}",
                f"${self.balance:,.2f}",
                f"${capital_invested:,.2f}",
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

        # Print performance table directly to console
        print("\n" + "=" * 65)
        print("📈 FINAL HOURLY SIMULATION PERFORMANCE")
        print("=" * 65)
        for _, row in summary_df.iterrows():
            print(f"  • {row['Metric']:<38}: {row['Value']}")
        print("=" * 65)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = output_dir / f"Simulation_Results_{revision}_{ts}.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            master_df.to_excel(writer, sheet_name='Master Trading Log', index=False)

        print(f"\n💾 Simulation Excel Ledger exported to:\n   👉 {excel_path}")


def main():
    print("=" * 70)
    print("🚀 HOURLY ASYMMETRIC ML BACKTESTING SIMULATION")
    print("=" * 70)

    config = load_settings()
    loader = MarketDataLoader(config)

    # 1. Load ML Model
    model_path = PROJECT_ROOT / config.get("paths", {}).get("model", "data/models/IE_Oracle_v6.0_Hourly.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"ML Model artifact not found at: {model_path}. Run train_hourly_model.py first.")

    print(f"🧠 Loading ML Oracle Brain: {model_path.name}")
    oracle_model = joblib.load(model_path)

    # 2. Score Prompt & Watchlist Sifting
    cfg_score = config.get("filters", {}).get("min_ticker_score", 3)
    score_in = input(f"👉 Minimum Ticker Score Cutoff [0-10] (Default {cfg_score}): ").strip()
    min_score = int(score_in) if score_in.isdigit() else cfg_score

    trap_thresh = float(config.get("filters", {}).get("trap_threshold", 0.35))
    whale_thresh = float(config.get("filters", {}).get("whale_threshold", 0.42))

    tickers, sector_map, earnings_df = loader.load_watchlist(min_score=min_score)
    print(f"📋 Watchlist Sifted: {len(tickers)} symbols active (Score >= {min_score})")

    # 3. Load Hourly Bars
    t0 = time.time()
    df_bars = loader.fetch_historical_bars(tickers=tickers, force_refresh=False)
    print(f"⚡ Loaded {len(df_bars):,} hourly candles in {time.time() - t0:.2f}s")

    # 4. Vectorized Indicator Calculations
    print("\n⚙️ Calculating Indicators across all symbols...")
    all_series = {}
    unique_tickers = df_bars.index.get_level_values("Ticker").unique()

    for t in unique_tickers:
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
        df['SL_L'] = np.floor(df['Lowest_Since'])

        df['L_Agg_Sig'] = (df['State_L_Agg'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (df['Close'] > df['EMA_200'])
        df['L_Tier_Sig'] = (df['State_L_Tier'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (df['Close'] > df['EMA_200'])

        all_series[t] = df

    # MultiIndex reassembly using keys
    full_df = pd.concat(all_series.values(), keys=all_series.keys(), names=['Ticker', 'Datetime'])

    obs_start = pd.to_datetime(config.get("simulation", {}).get("observation_start", "2021-06-01")).tz_localize("America/New_York")
    sim_df = full_df.loc[(slice(None), slice(obs_start, None)), :].swaplevel(0, 1).sort_index()
    unique_timestamps = sim_df.index.get_level_values(0).unique()

    print(f"⏳ Simulating chronologically across {len(unique_timestamps):,} hourly steps...")

    # 5. Chronological Simulation Loop
    portfolio = HourlyPortfolio(config, sector_map)

    for dt in unique_timestamps:
        bar_data = sim_df.loc[dt]

        # A. Position Exit Checks
        for t in list(portfolio.positions.keys()):
            if t not in bar_data.index:
                continue
            r = bar_data.loc[t]
            p = portfolio.positions[t]

            # Stop Loss Trigger
            if r['Low'] <= p['stop_loss']:
                portfolio.execute_sell(dt, t, p['stop_loss'], "Stop Loss Hit", r['RSI'])
            # Target Trigger (RSI 50 or ML-Assigned RSI 60 Whale Target)
            elif r['RSI'] >= p['target_rsi']:
                reason_str = f"RSI {p['target_rsi']} Whale Target" if p['target_rsi'] == 60 else "RSI 50 Target"
                portfolio.execute_sell(dt, t, r['Close'], reason_str, r['RSI'])

        # B. Setup Entry Scanning with ML Inference
        for t, r in bar_data.iterrows():
            if not portfolio.can_open_position(t, r['L_ID']):
                continue

            if r['L_Agg_Sig'] or r['L_Tier_Sig']:
                entry_price = r['Close']
                stop_loss = r['SL_L']
                if pd.isna(entry_price) or pd.isna(stop_loss) or entry_price <= stop_loss:
                    continue

                risk_tier = 0.02 if r['L_Agg_Sig'] else 0.005

                # Earnings proximity
                t_earnings = earnings_df[earnings_df['Ticker'] == t]['Earnings_Date']
                days_to_earn, days_since_earn = 45, 45
                if not t_earnings.empty:
                    deltas = (t_earnings.dt.tz_localize(dt.tz) - dt).dt.days
                    fut = deltas[deltas >= 0]
                    pst = deltas[deltas < 0]
                    if not fut.empty: days_to_earn = int(fut.min())
                    if not pst.empty: days_since_earn = int(abs(pst.max()))

                # Assemble ML Inference Vector
                feature_dict = {
                    'RSI_Extreme_Value': r['Extreme_RSI'],
                    'Dist_From_EMA_%': ((entry_price - r['EMA_200']) / r['EMA_200']) * 100,
                    'Entry_RVOL': r['RVOL'],
                    'Entry_ATR': r['ATR'],
                    'Days_To_Earnings': days_to_earn,
                    'Days_Since_Earnings': days_since_earn,
                    'Entry_Month': dt.month,
                    'Risk_Tier': risk_tier
                }
                feat_df = pd.DataFrame([feature_dict])[FEATURE_COLS].fillna(0)

                # Query ML Oracle
                probabilities = oracle_model.predict_proba(feat_df)[0]
                p_trap = probabilities[0]
                p_whale = probabilities[2] if len(probabilities) > 2 else 0.0

                # ML Veto Gate: Reject Traps
                if p_trap >= trap_thresh:
                    continue

                # ML Target Gate: Route to Whale (RSI 60) or Base Hit (RSI 50)
                assigned_target = 60 if p_whale >= whale_thresh else 50

                portfolio.execute_buy(
                    dt=dt,
                    ticker=t,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    extreme_rsi=r['Extreme_RSI'],
                    regime_id=r['L_ID'],
                    entry_ema=r['EMA_200'],
                    risk_tier=risk_tier,
                    entry_rvol=r['RVOL'],
                    entry_atr=r['ATR'],
                    target_rsi=assigned_target,
                    days_to_earn=days_to_earn,
                    days_since_earn=days_since_earn,
                    p_trap=p_trap,
                    p_whale=p_whale
                )

    # 6. Export Results
    output_dir = PROJECT_ROOT / config.get("paths", {}).get("output_dir", "SIM_Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio.export_results(output_dir, config.get("system", {}).get("revision", "2.0_Hourly_ML"))


if __name__ == "__main__":
    main()