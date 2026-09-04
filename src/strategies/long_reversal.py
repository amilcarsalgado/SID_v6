import numpy as np
import pandas as pd


def apply_regime_state_machine(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates stateful arming regime and entry/exit triggers."""
    df = df.copy()

    # Persistent State Machine (Armed when RSI < 35, Disarmed when RSI >= 50)
    df['State_L_Tier'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] >= 50, 0, np.nan))
    df['State_L_Tier'] = df['State_L_Tier'].ffill().fillna(0)

    # Regime groupings and low-water marks
    df['L_ID'] = ((df['State_L_Tier'] == 1) & (df['State_L_Tier'].shift(1) == 0)).cumsum()
    valid_lows = df['Low'].where(df['State_L_Tier'] == 1, np.nan)
    df['Lowest_Since'] = valid_lows.groupby(df['L_ID']).cummin()

    valid_rsi = df['RSI'].where(df['State_L_Tier'] == 1, np.nan)
    df['Extreme_RSI'] = valid_rsi.groupby(df['L_ID']).cummin()

    # Protected Stop Loss Floor
    df['SL_L'] = np.floor(df['Lowest_Since'])

    # Entry Triggers: Armed state + 3-bar RSI hook window + MACD hook + Trend Confirmation
    df['L_Tier_Sig'] = (
            (df['State_L_Tier'] == 1) &
            (df['R_Hook_U'].rolling(3).max() == 1) &
            df['M_Hook_U'] &
            (df['Close'] > df['EMA_200'])
    )
    return df