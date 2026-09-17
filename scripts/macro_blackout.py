"""
SID Equities v7.0 - Macroeconomic Blackout Parser
=================================================
Author: Alvaro Salgado

Operational Directives & Functional Architecture:
-------------------------------------------------
1. Dynamic Buffer Window:
   Parses event timestamps ('datetime_utc') and dynamically builds the safety
   corridor using 'buffer_minutes_before' and 'buffer_minutes_after'.

2. Asset Alignment (Blast Radius):
   Maps 'USD' and 'ALL_FX' in the JSON directly to the 'EQUITIES' engine,
   ensuring US macro catalysts (FOMC, NFP, CPI, PPI) halt equity entries.

3. Stop-Losses & Exits Remain Active:
   The blackout strictly vetoes *new entries*. Resting OTO broker-side
   stop-losses and RSI profit-harvesting loops remain fully operational.

4. Time Mechanics:
   Strict UTC comparison via datetime.timezone.utc to prevent timezone drift.
"""

import json
import logging
from datetime import datetime, timedelta, timezone


class MacroBlackoutParser:
    """
    Parses blackout_dates.json using event timestamps and dynamic before/after buffers.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.logger = logging.getLogger("SID_Equities.MacroBlackout")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.blackout_events = self._load_calendar()

    def _load_calendar(self) -> list:
        try:
            with open(self.json_path, 'r') as file:
                data = json.load(file)
                # Matches your exact top-level JSON key: 'blackout_events'
                events = data.get("blackout_events", [])
                self.logger.info(f"Loaded {len(events)} macro blackout events from calendar.")
                return events
        except FileNotFoundError:
            self.logger.error(f"CRITICAL: Blackout calendar not found at {self.json_path}.")
            return []
        except json.JSONDecodeError:
            self.logger.error("CRITICAL: Invalid JSON format in blackout calendar.")
            return []

    def is_trade_allowed(self, current_time_utc: datetime, asset_type: str = "EQUITIES") -> bool:
        """
        Checks if current_time_utc falls inside any active macro buffer window.
        """
        # Ensure UTC timezone alignment
        if current_time_utc.tzinfo is None:
            current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
        elif current_time_utc.tzinfo != timezone.utc:
            current_time_utc = current_time_utc.astimezone(timezone.utc)

        for event in self.blackout_events:
            currencies = event.get("currency", [])

            # Determine relevance for the active engine
            # For Equities: Any USD, CRYPTO, or ALL_FX event impacts the engine
            is_relevant = False
            if asset_type == "EQUITIES":
                is_relevant = any(c in ["USD", "ALL_FX", "EQUITIES"] for c in currencies)
            else:
                # Direct match for Forex currency pairs (e.g. 'EUR', 'GBP')
                is_relevant = (asset_type in currencies) or ("ALL_FX" in currencies)

            if not is_relevant:
                continue

            try:
                # Parse event base time: e.g. "2026-09-16T18:00:00Z"
                raw_time_str = event["datetime_utc"].replace("Z", "+00:00")
                event_time = datetime.fromisoformat(raw_time_str)

                # Dynamically construct window from buffer minutes
                mins_before = int(event.get("buffer_minutes_before", 0))
                mins_after = int(event.get("buffer_minutes_after", 0))

                start_window = event_time - timedelta(minutes=mins_before)
                end_window = event_time + timedelta(minutes=mins_after)

                # Check if current time falls within active buffer
                if start_window <= current_time_utc <= end_window:
                    self.logger.info(
                        f"🛡️ MACRO VETO: [{event.get('event')}] Active! "
                        f"Window: {start_window.strftime('%H:%M')} - {end_window.strftime('%H:%M')} UTC. "
                        f"New entries halted."
                    )
                    return False

            except Exception as e:
                self.logger.error(f"Error evaluating event '{event.get('event')}': {e}")
                continue

        return True