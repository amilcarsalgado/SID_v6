import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Applies standardized technical indicators to an OHLCV DataFrame."""
    df = df.copy()

    # 200 Exponential Moving Average
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Normalized Relative Volume (20-bar)
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Average True Range (14-period)
    tr = np.maximum(
        (df['High'] - df['Low']),
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = tr.rolling(14).mean()

    # Wilder-smoothed 14-period RSI
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    df['MACD_Line'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    # Reversal Hooks
    df['R_Hook_U'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) <= df['RSI'].shift(2))
    df['M_Hook_U'] = (df['MACD_Line'] > df['MACD_Line'].shift(1)) & (
                df['MACD_Line'].shift(1) <= df['MACD_Line'].shift(2))

    return df