"""
===============================================================================
SYSTEM: VECTORIZED CHRONOLOGICAL BACKTESTING & PORTFOLIO SIMULATOR
LOCATION: src/trading_engine/engine/simulator.py
VERSION: 2.7_ML (Modular Architecture Edition)
===============================================================================
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class VectorSimulator:
    def __init__(self, data: pd.DataFrame, sector_map: dict, earnings_df: pd.DataFrame, config: dict):
        self.data = data
        self.sector_map = sector_map if sector_map else {}
        self.earnings_df = earnings_df
        self.config = config

        # Resolve paths from config
        self.export_dir = PROJECT_ROOT / str(config.get("paths", {}).get("outputs_dir", "SIM_Results")).strip("'\" ")
        self.export_dir.mkdir(parents=True, exist_ok=True)

        model_rel_path = str(config.get("paths", {}).get("model_file", "data/models/IE_Oracle_v5.2.joblib")).strip(
            "'\" ")
        self.model_path = PROJECT_ROOT / model_rel_path
        if not self.model_path.exists():
            # Fallback to local working root if specified directly
            self.model_path = PROJECT_ROOT / "IE_Oracle_v5.2.joblib"

        # Load ML Brain
        self.ai_oracle = self._load_model()
        self.trap_threshold = config.get("model", {}).get("trap_threshold", 0.35)
        self.whale_threshold = config.get("model", {}).get("whale_threshold", 0.42)

        # Simulation Portfolio Variables
        sim_cfg = config.get("simulation", {})
        self.initial_balance = float(sim_cfg.get("initial_balance", 50000.0))
        self.balance = self.initial_balance
        self.current_equity = self.initial_balance
        self.commission = float(sim_cfg.get("commission", 1.00))

        self.sizing_risk_base = float(sim_cfg.get("sizing_risk_base", 20000.0))
        self.max_open_positions = int(sim_cfg.get("max_open_positions", 15))
        self.min_score_threshold = int(config.get("filters", {}).get("min_ticker_score", 0))

        # Risk Management Governors
        self.high_water_mark = self.initial_balance
        self.defensive_risk_pct = float(sim_cfg.get("defensive_risk_pct", 0.01))
        self.drawdown_threshold = float(sim_cfg.get("drawdown_threshold", 0.05))

        # Ledger & Positions
        self.positions = {}
        self.master_log = []
        self.burned_long_regimes = {}

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            print(f"🧠 AI ORACLE INITIALIZATION SUCCESSFUL: Loaded '{self.model_path.name}'")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL ERROR: Could not load AI Brain model from {self.model_path} ({e})")

    def query_ai_oracle(self, feature_dict: dict) -> int:
        feature_vector = np.array([[
            feature_dict['RSI_Extreme_Value'],
            feature_dict['Dist_From_EMA_%'],
            feature_dict['Entry_RVOL'],
            feature_dict['Entry_ATR'],
            feature_dict['Days_To_Earnings'],
            feature_dict['Days_Since_Earnings'],
            feature_dict['Entry_Month'],
            feature_dict['Risk_Tier']
        ]])

        probs = self.ai_oracle.predict_proba(feature_vector)[0]
        if probs[0] > self.trap_threshold:
            return 0  # ABORT (Trap)
        elif probs[2] > self.whale_threshold:
            return 2  # Target RSI 60 (Whale)
        else:
            return 1  # Target RSI 50 (Base Hit)

    def execute_buy(self, date, ticker, entry_price, stop_loss, direction, extreme_date,
                    extreme_price, extreme_rsi, regime_id, entry_ema, risk_tier, entry_rvol, entry_atr,
                    days_to_earnings, days_since_earnings):
        if pd.isna(entry_price) or pd.isna(stop_loss) or entry_price == stop_loss:
            return
        if direction == 'LONG' and regime_id in self.burned_long_regimes.get(ticker, []):
            return

        self.high_water_mark = max(self.high_water_mark, self.current_equity)
        current_drawdown = (self.high_water_mark - self.current_equity) / self.high_water_mark

        is_governor_active = current_drawdown >= self.drawdown_threshold
        active_risk_pct = self.defensive_risk_pct if is_governor_active else risk_tier

        ai_payload = {
            'RSI_Extreme_Value': extreme_rsi,
            'Dist_From_EMA_%': ((entry_price - entry_ema) / entry_ema) * 100,
            'Entry_RVOL': entry_rvol,
            'Entry_ATR': entry_atr,
            'Days_To_Earnings': days_to_earnings,
            'Days_Since_Earnings': days_since_earnings,
            'Entry_Month': date.month,
            'Risk_Tier': active_risk_pct
        }

        ai_decision = self.query_ai_oracle(ai_payload)
        if ai_decision == 0:
            return

        exit_rule = "RSI_50_TARGET" if ai_decision == 1 else "RSI_60_WHALE"

        dollar_risk = self.sizing_risk_base * active_risk_pct
        risk_per_share = abs(entry_price - stop_loss)
        max_shares = int(dollar_risk / risk_per_share)
        position_size = max_shares * entry_price

        is_ceiling_breached = len(self.positions) >= self.max_open_positions
        is_cash_insufficient = (position_size + self.commission) > self.balance

        if max_shares <= 0 or is_ceiling_breached or is_cash_insufficient:
            invested_at_choke = sum(p['invested_capital'] for p in self.positions.values())
            status_label = 'CS_Ceiling' if is_ceiling_breached else 'CS_Liquidity'
            reason = f"Position Ceiling Active ({self.max_open_positions} Max)" if is_ceiling_breached else 'Insufficient Liquidity'

            self.master_log.append({
                'Ticker': ticker, 'Sector_Name': self.sector_map.get(ticker, 'Unmapped'),
                'Direction': direction, 'Applied_Risk_Pct': f"{active_risk_pct * 100}%",
                'RSI_Extreme_Value': round(extreme_rsi, 2), 'Entry_Date': str(date.date()),
                'Entry_Price': round(entry_price, 2), 'Entry_EMA_200': round(entry_ema, 2),
                'Dist_From_EMA_%': round(((entry_price - entry_ema) / entry_ema) * 100, 2),
                'Entry_RVOL': round(entry_rvol, 2), 'Entry_ATR': round(entry_atr, 2),
                'Exit_Date': str(date.date()), 'Hold_Duration': 0, 'Exit_Price': 0.00, 'Exit_RSI': 0.00,
                'AI_Regime_Assigned': ai_decision, 'Exit_Reason': reason,
                'Net_PnL (After Fees)': 0.00, 'Account_Equity': round(self.current_equity, 2),
                'Remaining_Available_Cash': round(self.balance, 2),
                'Invested_Capital_Snapshot': round(invested_at_choke, 2),
                'Execution_Status': status_label
            })
            return

        self.balance -= (position_size + self.commission)
        self.positions[ticker] = {
            'entry_date': date, 'direction': direction, 'shares': max_shares,
            'entry_price': entry_price, 'invested_capital': position_size,
            'stop_loss': stop_loss, 'extreme_date': extreme_date,
            'extreme_price': extreme_price, 'extreme_rsi': extreme_rsi,
            'regime_id': regime_id, 'applied_risk': active_risk_pct,
            'entry_ema': entry_ema, 'entry_rvol': entry_rvol, 'entry_atr': entry_atr,
            'ai_regime': ai_decision, 'exit_rule': exit_rule
        }
        self.burned_long_regimes.setdefault(ticker, []).append(regime_id)

    def execute_sell(self, date, ticker, exit_price, reason, exit_rsi):
        pos = self.positions.pop(ticker)
        gain_per_share = (exit_price - pos['entry_price'])
        total_profit = (gain_per_share * pos['shares']) - (self.commission * 2)

        cash_returned = pos['shares'] * exit_price
        self.balance += (cash_returned - self.commission)

        invested_at_exit = sum(p['invested_capital'] for p in self.positions.values())
        self.current_equity = self.balance + invested_at_exit
        self.high_water_mark = max(self.high_water_mark, self.current_equity)

        self.master_log.append({
            'Ticker': ticker,
            'Sector_Name': self.sector_map.get(ticker, 'Unmapped'),
            'Direction': pos['direction'],
            'Applied_Risk_Pct': f"{pos['applied_risk'] * 100}%",
            'RSI_Extreme_Value': round(pos['extreme_rsi'], 2),
            'Entry_Date': str(pos['entry_date'].date()),
            'Entry_Price': round(pos['entry_price'], 2),
            'Entry_EMA_200': round(pos['entry_ema'], 2),
            'Dist_From_EMA_%': round(((pos['entry_price'] - pos['entry_ema']) / pos['entry_ema']) * 100, 2),
            'Entry_RVOL': round(pos['entry_rvol'], 2),
            'Entry_ATR': round(pos['entry_atr'], 2),
            'Exit_Date': str(date.date()),
            'Hold_Duration': (date - pos['entry_date']).days,
            'Exit_Price': round(exit_price, 2),
            'Exit_RSI': round(exit_rsi, 2),
            'AI_Regime_Assigned': pos['ai_regime'],
            'Exit_Reason': reason,
            'Net_PnL (After Fees)': round(total_profit, 2),
            'Account_Equity': round(self.current_equity, 2),
            'Remaining_Available_Cash': round(self.balance, 2),
            'Invested_Capital_Snapshot': round(invested_at_exit, 2),
            'Execution_Status': 'EXEC'
        })

    def run(self) -> dict:
        """Executes indicator calculations and chronological timeline simulation."""
        tickers = self.data.index.get_level_values('Ticker').unique().tolist()
        all_series = []

        print(f"📊 Vectorizing indicators across {len(tickers)} tickers...")
        for t in tickers:
            df = self.data.xs(t, level='Ticker').copy()
            df.dropna(how='all', inplace=True)

            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
            tr = np.maximum((df['High'] - df['Low']),
                            np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                       abs(df['Low'] - df['Close'].shift(1))))
            df['ATR'] = tr.rolling(14).mean()

            delta = df['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            df['RSI'] = 100 - (100 / (1 + up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()))
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
            df['Extreme_Date'] = df.index.to_series().where(df['State_L_Tier'] == 1, pd.NaT).groupby(
                df['L_ID']).transform('first')
            df['SL_L'] = np.floor(df['Lowest_Since'])

            df['L_Agg_Sig'] = (df['State_L_Agg'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (
                        df['Close'] > df['EMA_200'])
            df['L_Tier_Sig'] = (df['State_L_Tier'] == 1) & (df['R_Hook_U'].rolling(3).max() == 1) & df['M_Hook_U'] & (
                        df['Close'] > df['EMA_200'])
            all_series.append(df)

        full_df = pd.concat(all_series, keys=tickers, names=['Ticker', 'Datetime'])
        obs_start = pd.to_datetime(
            self.config.get("simulation", {}).get("observation_start", "2021-06-01")).tz_localize("America/New_York")

        sim_df = full_df.loc[(slice(None), slice(obs_start, None)), :].swaplevel(0, 1).sort_index()
        unique_datetimes = sim_df.index.get_level_values('Datetime').unique()

        print(f"⏳ Stepping through {len(unique_datetimes):,} timeline timestamps...")
        for dt in unique_datetimes:
            step_data = sim_df.loc[dt]

            # 1. EVALUATE EXITS
            for t in list(self.positions.keys()):
                if t not in step_data.index: continue
                r, p = step_data.loc[t], self.positions[t]

                if r['Low'] <= p['stop_loss']:
                    self.execute_sell(dt, t, p['stop_loss'], "Stop Loss Hit", r['RSI'])
                elif p['exit_rule'] == "RSI_50_TARGET" and r['RSI'] >= 50:
                    self.execute_sell(dt, t, r['Close'], "RSI 50 Target Met", r['RSI'])
                elif p['exit_rule'] == "RSI_60_WHALE" and r['RSI'] >= 60:
                    self.execute_sell(dt, t, r['Close'], "RSI 60 Whale Target Met", r['RSI'])

            # 2. EVALUATE ENTRIES
            for t, r in step_data.iterrows():
                if t not in self.positions:
                    if r['L_Agg_Sig'] or r['L_Tier_Sig']:
                        risk_tier = 0.02 if r['L_Agg_Sig'] else 0.005

                        t_earnings = self.earnings_df[self.earnings_df['Ticker'] == t]['Earnings_Date']
                        days_to_earnings, days_since_earnings = 45, 45
                        if not t_earnings.empty:
                            deltas = (t_earnings.dt.tz_localize(dt.tz) - dt).dt.days
                            future_deltas = deltas[deltas >= 0]
                            past_deltas = deltas[deltas < 0]
                            if not future_deltas.empty: days_to_earnings = future_deltas.min()
                            if not past_deltas.empty: days_since_earnings = abs(past_deltas.max())

                        self.execute_buy(
                            dt, t, r['Close'], r['SL_L'], 'LONG', r['Extreme_Date'],
                            r['Lowest_Since'], r['Extreme_RSI'], r['L_ID'], r['EMA_200'],
                            risk_tier, r['RVOL'], r['ATR'], days_to_earnings, days_since_earnings
                        )

        # Export and return metrics
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"Simulation_Results_{ts}.xlsx"
        return self.export_ledger(unique_datetimes, export_file)

    def export_ledger(self, unique_timestamps, filename: str) -> dict:
        if not self.master_log:
            print("⚠️ No trades or choke events generated during this backtest run.")
            return {}

        master_df = pd.DataFrame(self.master_log).sort_values(by=['Entry_Date', 'Ticker'])
        executed_trades = master_df[master_df['Execution_Status'] == 'EXEC']
        total_liquidity_chokes = len(master_df[master_df['Execution_Status'] == 'CS_Liquidity'])
        total_ceiling_chokes = len(master_df[master_df['Execution_Status'] == 'CS_Ceiling'])

        total_trades = len(executed_trades)
        wins = executed_trades[executed_trades['Net_PnL (After Fees)'] > 0]
        losses = executed_trades[executed_trades['Net_PnL (After Fees)'] <= 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = wins['Net_PnL (After Fees)'].sum()
        gross_loss = abs(losses['Net_PnL (After Fees)'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        capital_invested = sum(p['invested_capital'] for p in self.positions.values())
        total_return = ((self.current_equity - self.initial_balance) / self.initial_balance) * 100

        summary_metrics = {
            "Total Closed Trades": total_trades,
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Initial Account Value": f"${self.initial_balance:,.2f}",
            "Final Account Equity": f"${self.current_equity:,.2f}",
            "Total Return (%)": f"{total_return:.2f}%",
            "Winning Trades": len(wins),
            "Losing Trades": len(losses),
            "Choked: Liquidity": total_liquidity_chokes,
            "Choked: Position Ceiling": total_ceiling_chokes,
        }

        target_path = self.export_dir / filename
        with pd.ExcelWriter(target_path, engine='openpyxl') as writer:
            pd.DataFrame(list(summary_metrics.items()), columns=["Metric", "Value"]).to_excel(writer,
                                                                                              sheet_name='Summary',
                                                                                              index=False)
            master_df.to_excel(writer, sheet_name='Master Trading Log', index=False)

        print(f"\n📂 Export Generated: Workbook written to -> {target_path}")
        return summary_metrics

    def print_metrics(self):
        """Standard summary printing for console display."""
        pass