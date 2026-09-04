import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import os
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="Autonomous Trading Autopsy", layout="wide")


@st.cache_resource
def get_clients():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)
    return trading_client, data_client


def fetch_bar_data(data_client, ticker, entry_time):
    try:
        parsed_entry_dt = pd.to_datetime(entry_time) if entry_time != 'N/A' else pd.Timestamp.now() - pd.Timedelta(
            days=15)
        start_dt = parsed_entry_dt - pd.Timedelta(days=30)
    except Exception:
        start_dt = pd.Timestamp.now() - pd.Timedelta(days=30)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Hour,
        start=start_dt,
        feed=DataFeed.IEX
    )

    try:
        df = data_client.get_stock_bars(request).df
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Resilient index parsing across Alpaca SDK variations
    if isinstance(df.index, pd.MultiIndex):
        level_names = [str(name).lower() for name in df.index.names]
        if "ticker" in level_names:
            level_idx = level_names.index("ticker")
            actual_level_name = df.index.names[level_idx]
            if ticker in df.index.get_level_values(actual_level_name):
                df = df.xs(ticker, level=actual_level_name).reset_index()
            else:
                return None
        else:
            df = df.reset_index()
    else:
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker]
        elif "ticker" in df.columns:
            df = df[df["ticker"] == ticker]
        df = df.reset_index(drop=True)

    if df.empty:
        return None

    if 'timestamp' in df.columns:
        df['datetime'] = df['timestamp'].dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time']).dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        return None

    return df


def calculate_indicators(df):
    df['SMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


def load_terminated_trades():
    results_dir = PROJECT_ROOT / "SIM_Results"
    terminated_list = []

    if results_dir.exists():
        for file in sorted(results_dir.glob("*.xlsx"), reverse=True):
            try:
                xls = pd.ExcelFile(file)
                sheet_to_use = "Closed_Trades" if "Closed_Trades" in xls.sheet_names else (
                    "Closed Trades Log" if "Closed Trades Log" in xls.sheet_names else None)
                if not sheet_to_use:
                    continue

                df = pd.read_excel(file, sheet_name=sheet_to_use)
                for _, row in df.iterrows():
                    entry_t = row.get('Entry_Time') or row.get('Entry_Time (EST)') or 'N/A'
                    exit_t = row.get('Exit_Time') or row.get('Exit_Time (EST)') or 'N/A'
                    avg_e = row.get('Avg_Entry') if 'Avg_Entry' in row else row.get('Avg_Entry_Price', 0.0)
                    exit_p = row.get('Exit_Price', 0.0)
                    sl_p = row.get('SL_Price', 0.0)

                    terminated_list.append({
                        'ticker': str(row.get('Ticker', 'UNKNOWN')),
                        'entry_time': str(entry_t),
                        'exit_time': str(exit_t),
                        'entry_price': float(avg_e),
                        'exit_price': float(exit_p),
                        'stop_price': float(sl_p)
                    })
            except Exception:
                continue

    return terminated_list


# Streamlit UI Configuration
st.title("📈 Autonomous Trading Engine — Visual Autopsy Dashboard")

trading_client, data_client = get_clients()

state_file = PROJECT_ROOT / "data" / "alpaca_state.json"
state_meta = {}
if state_file.exists():
    try:
        with open(state_file, "r") as f:
            state_meta = json.load(f).get("position_meta", {})
    except Exception:
        state_meta = {}

st.sidebar.header("Trade Selector")
trade_type = st.sidebar.radio("Select Category", ["Active Positions", "Terminated Trades"])

selected_trade = None

if trade_type == "Active Positions":
    try:
        open_positions = trading_client.get_all_positions()
    except Exception as e:
        st.sidebar.error(f"Error querying Alpaca: {e}")
        open_positions = []

    if not open_positions:
        st.sidebar.info("No active open positions on Alpaca.")
    else:
        active_options = {p.symbol: p for p in open_positions}
        chosen_ticker = st.sidebar.selectbox("Choose Ticker", list(active_options.keys()))
        p = active_options[chosen_ticker]
        meta = state_meta.get(chosen_ticker, {})

        avg_entry = float(meta.get('avg_entry', p.avg_entry_price))
        sl_price = float(meta.get('sl_price', avg_entry * 0.95))

        selected_trade = {
            'ticker': chosen_ticker,
            'entry_time': str(meta.get('entry_time', 'N/A')),
            'exit_time': '',
            'entry_price': avg_entry,
            'exit_price': 0.0,
            'stop_price': sl_price,
            'is_active': True
        }

else:
    terminated_list = load_terminated_trades()
    if not terminated_list:
        st.sidebar.info("No closed trades recorded yet in SIM_Results.")
    else:
        term_options = {f"{tr['ticker']} | In: {tr['entry_time'][:16]}": tr for tr in terminated_list}
        chosen_label = st.sidebar.selectbox("Choose Terminated Trade", list(term_options.keys()))
        selected_trade = term_options[chosen_label]
        selected_trade['is_active'] = False

if selected_trade:
    ticker = selected_trade['ticker']
    mode_label = "ACTIVE" if selected_trade['is_active'] else "TERMINATED"
    st.subheader(f"Trade Autopsy: {ticker} [{mode_label}]")

    df = fetch_bar_data(data_client, ticker, selected_trade['entry_time'])
    if df is not None and not df.empty:
        df = calculate_indicators(df)
        df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.55, 0.22, 0.23]
        )

        # Price Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df['date_str'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        ), row=1, col=1)

        # 200 EMA / SMA Trend Line
        fig.add_trace(go.Scatter(
            x=df['date_str'], y=df['SMA_200'],
            line=dict(color='orange', width=1.2), name='EMA 200'
        ), row=1, col=1)

        # Stop-Loss Reference Level
        if selected_trade['stop_price'] > 0:
            fig.add_hline(
                y=selected_trade['stop_price'], line_dash="dash", line_color="red",
                annotation_text=f"Stop: ${selected_trade['stop_price']:.2f}", row=1, col=1
            )

        # Entry Marker Coordinate Resolution
        entry_t = selected_trade['entry_time']
        entry_dt = pd.to_datetime(entry_t) if entry_t != 'N/A' else None
        if entry_dt is not None:
            matched = df[df['datetime'] >= entry_dt]
            entry_x = matched['date_str'].iloc[0] if not matched.empty else df['date_str'].iloc[0]
        else:
            entry_x = df['date_str'].iloc[0]

        fig.add_trace(go.Scatter(
            x=[entry_x], y=[selected_trade['entry_price']],
            mode='markers+text', marker=dict(symbol='triangle-up', size=14, color='#00e676'),
            text=[f"BUY @ ${selected_trade['entry_price']:.2f}"], textposition="bottom center", name='Entry'
        ), row=1, col=1)

        # Exit Marker (For Terminated Positions)
        if not selected_trade['is_active'] and selected_trade['exit_time'] != 'N/A':
            exit_dt = pd.to_datetime(selected_trade['exit_time'])
            matched_exit = df[df['datetime'] >= exit_dt]
            exit_x = matched_exit['date_str'].iloc[0] if not matched_exit.empty else df['date_str'].iloc[-1]

            fig.add_trace(go.Scatter(
                x=[exit_x], y=[selected_trade['exit_price']],
                mode='markers+text', marker=dict(symbol='triangle-down', size=14, color='#2979ff'),
                text=[f"EXIT @ ${selected_trade['exit_price']:.2f}"], textposition="top center", name='Exit'
            ), row=1, col=1)

        # Subplot 2: RSI
        fig.add_trace(go.Scatter(
            x=df['date_str'], y=df['RSI'],
            line=dict(color='#26a69a', width=1.5), name='RSI (14)'
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#787b86", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#787b86", row=2, col=1)

        # Subplot 3: MACD
        fig.add_trace(go.Scatter(
            x=df['date_str'], y=df['MACD'],
            line=dict(color='#2962ff', width=1.5), name='MACD'
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df['date_str'], y=df['MACD_Signal'],
            line=dict(color='#ff6d00', width=1.5), name='Signal'
        ), row=3, col=1)

        fig.update_layout(
            height=850,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            xaxis_type='category',
            xaxis3=dict(type='category', nticks=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not retrieve historical bars for {ticker}.")

'''
streamlit run scripts/autopsy_app.py

'''