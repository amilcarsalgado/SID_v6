"""
SID Equities v7.0 - Macroeconomic Blackout Parser
=================================================
Author: Alvaro Salgado

This module acts as the first line of defense for the autonomous trading engine,
enforcing strict risk boundaries around Tier-1 macroeconomic volatility events.

Operational Directives & Functional Architecture:
-------------------------------------------------
1. Asset-Specific Blast Radius:
   The parser isolates macro events by asset tags. A European Central Bank (ECB)
   rate decision tagged 'EUR' will halt EUR Forex crosses, but the Equities
   daemon (trading 'EQUITIES' or 'USD') will continue scanning seamlessly.
   Events tracked include central bank actions (FOMC, BOE, BOJ, BOC),
   US indicators (CPI, PPI, Non-Farm Payrolls), and procedural events
   like the US Senate CLARITY Act vote.

2. Pre-Event Liquidity Vacuum (The Veto):
   Institutional market makers pull liquidity 30-60 minutes prior to major
   announcements. This creates a thin order book, spreading bid-ask widths,
   and causing erratic "pre-event chop". The parser intentionally vetoes new
   technical setups during this window to prevent the engine from buying into
   false RSI/MACD hooks or suffering catastrophic slippage.

3. Stop-Losses & Exits Remain Active:
   The blackout strictly applies to *opening new positions*.
   - Preset OTO (One-Triggers-Other) stop-losses resting on the broker's side
     remain fully active to protect capital during market dumps.
   - The engine's hourly scan continues to monitor for take-profit triggers.
     If a position hits its RSI target during the blackout, the exit executes.

4. Resumption of Operations:
   Once the event's UTC `end_time` has expired (the post-event safety buffer),
   the system resumes normal scanning for genuine momentum trends.

5. Time Mechanics:
   The module strictly coerces all comparisons to UTC. Local timezone handling
   (like EST/EDT shifts) is completely bypassed to prevent daylight saving errors.

6. Breakout Algorithm:

https://www.youtube.com/watch?v=oyuyeYi_7rw

Video Summary: Automated Price Break Out Detection in Python
In this tutorial, the creator (CodeTrading) walks through how to write a Python algorithm that automatically detects price breakouts. The script looks for a pattern where an asset bounces at least three times off a specific price level (a key support or resistance zone). It includes logic to define a "Zone width" so the bounces don't have to perfectly align to the exact penny.

When a candle officially breaks and closes above or below this key zone, the script flags it as a valid bullish or bearish breakout signal. It also specifically covers how to implement a "Gap window" in your Pandas dataframe slicing to avoid look-ahead bias (accidentally allowing the algorithm to peek into the future when backtesting). This could be a very useful structural reference if you decide to refine the entry logic for SID Equities.

"""

import json
import logging
from datetime import datetime, timezone


class MacroBlackoutParser:
    """
    Loads a JSON calendar of macro events and validates if current execution
    is permitted based on asset type and pre-defined time buffers.
    """

    def __init__(self, json_path: str):
        """
        Initializes the blackout parser and loads the macro calendar into memory.

        :param json_path: Path to the blackout_dates.json file.
        """
        self.json_path = json_path
        self.blackout_events = self._load_calendar()

        # Configure a local logger for the blackout module
        self.logger = logging.getLogger("SID_Equities.MacroBlackout")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _load_calendar(self) -> list:
        """
        Loads and parses the JSON calendar into memory.

        Returns an empty list and defaults to a safe-mode state if the file
        is missing or improperly formatted.
        """
        try:
            with open(self.json_path, 'r') as file:
                data = json.load(file)
                return data.get("events", [])
        except FileNotFoundError:
            logging.error(
                f"CRITICAL: Blackout calendar not found at {self.json_path}. Defaulting to safe-mode (empty calendar).")
            return []
        except json.JSONDecodeError:
            logging.error("CRITICAL: Invalid JSON format in blackout calendar.")
            return []

    def is_trade_allowed(self, current_time_utc: datetime, asset_type: str) -> bool:
        """
        Validates if the current timestamp falls outside an active blackout window.

        Execution Rules:
        - Checks the asset_type against the event's impacted_assets array.
        - Coerces the current_time_utc to strict UTC to avoid timezone drift.
        - Vetoes new entry scans if current_time_utc is within the buffer window.
        - Allows take-profit exits and relies on existing OTO stop-losses
          if the market drops during the event.

        :param current_time_utc: A timezone-aware datetime object in UTC.
        :param asset_type: Asset class tag (e.g., 'EQUITIES', 'USD', 'EUR').
        :return: True if safe to open new trades, False if vetoed by a macro event.
        """
        # Ensure the passed datetime is strictly UTC to prevent timezone drift
        if current_time_utc.tzinfo is None or current_time_utc.tzinfo != timezone.utc:
            self.logger.warning("Timestamp passed to blackout parser lacks UTC tzinfo. Coercing to UTC.")
            current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)

        for event in self.blackout_events:
            # Check if the event impacts the asset the daemon is trading
            if asset_type in event.get("impacted_assets", []):

                try:
                    # Parse JSON UTC strings into datetime objects using fromisoformat
                    start_str = event["start_time_utc"].replace("Z", "+00:00")
                    end_str = event["end_time_utc"].replace("Z", "+00:00")

                    start_window = datetime.fromisoformat(start_str)
                    end_window = datetime.fromisoformat(end_str)

                    # Determine if the current time falls within the buffer window
                    if start_window <= current_time_utc <= end_window:
                        self.logger.info(
                            f"🛡️ MACRO VETO: Trading halted for {asset_type} due to active event: {event['event_name']}"
                        )
                        return False  # VETO - Do not open new trades

                except ValueError as e:
                    self.logger.error(f"Error parsing timestamps for event '{event.get('event_name')}': {e}")
                    continue

        return True  # Safe to execute new entries