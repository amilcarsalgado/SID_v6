import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Official Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Project root resolution (4 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class MarketDataLoader:
    def __init__(self, config: dict):
        """
        Initializes the data loader with absolute project root path resolution
        and Alpaca client credentials.
        """
        self.config = config

        # Resolve cache directory relative to project root
        raw_cache_path = str(config.get("paths", {}).get("cache_dir", "data/cache")).strip("'\" ")
        self.cache_dir = (PROJECT_ROOT / raw_cache_path).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve watchlist path relative to project root
        raw_wl_path = str(config.get("paths", {}).get("watchlist", "config/symbols.xlsx")).strip("'\" ")
        self.watchlist_path = (PROJECT_ROOT / raw_wl_path).resolve()

        # Load environment variables from the project root .env file
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            print("⚠️ [WARNING] Alpaca API credentials not found in environment (.env). "
                  "Cached data will load fine, but new API downloads will fail.", flush=True)
            self.client = None
        else:
            self.client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def get_cache_file(self, timeframe: TimeFrame = TimeFrame.Hour) -> Path:
        """Returns the canonical absolute path to the pickle cache file."""
        tf_label = "hourly" if timeframe == TimeFrame.Hour else "daily"
        return (self.cache_dir / f"market_data_{tf_label}.pkl").resolve()

    def cache_exists(self, timeframe: TimeFrame = TimeFrame.Hour) -> bool:
        """Checks if a valid, non-empty cache file exists on disk."""
        cf = self.get_cache_file(timeframe)
        return cf.exists() and cf.is_file() and cf.stat().st_size > 0

    def get_cached_tickers(self, timeframe: TimeFrame = TimeFrame.Hour) -> Set[str]:
        """Returns the set of ticker symbols currently stored in the cache."""
        if not self.cache_exists(timeframe):
            return set()
        try:
            cached_df = pd.read_pickle(self.get_cache_file(timeframe))
            return set(cached_df.index.get_level_values("Ticker").unique())
        except Exception:
            return set()

    def load_watchlist(self, min_score: int = 0) -> Tuple[List[str], Dict[str, str], pd.DataFrame]:
        """
        Parses config/symbols.xlsx, filters by Enabled and Column G (Score) cutoff,
        and returns tickers, sector map, and the earnings calendar dataframe.
        """
        if not self.watchlist_path.exists():
            raise FileNotFoundError(f"Watchlist file not found at: {self.watchlist_path}")

        wl_df = pd.read_excel(self.watchlist_path, sheet_name="Main")
        wl_df["Ticker"] = wl_df["Ticker"].astype(str).str.strip().str.upper()

        # Enforce enabled flag
        valid_true = [True, 1, "1", "YES", "Y", "TRUE", "T", "ENABLE", "ENABLED"]
        enabled_mask = wl_df["Enabled"].apply(
            lambda v: isinstance(v, bool) and v or str(v).strip().upper() in valid_true
        )
        filtered_df = wl_df[enabled_mask].copy()

        # Target Column G (Score)
        if "Score" in filtered_df.columns:
            filtered_df["Score"] = pd.to_numeric(filtered_df["Score"], errors="coerce").fillna(0)
        elif "Ticker_Score" in filtered_df.columns:
            filtered_df["Score"] = pd.to_numeric(filtered_df["Ticker_Score"], errors="coerce").fillna(0)
        else:
            score_col_name = filtered_df.columns[6]
            filtered_df["Score"] = pd.to_numeric(filtered_df[score_col_name], errors="coerce").fillna(0)

        # Apply minimum score filter
        final_df = filtered_df[filtered_df["Score"] >= min_score]
        tickers = final_df["Ticker"].tolist()
        sector_map = dict(zip(final_df["Ticker"], final_df.get("Sector_Name", "Unmapped")))

        # Load Earnings sheet if present
        try:
            earnings_df = pd.read_excel(self.watchlist_path, sheet_name="Earnings")
            earnings_df["Earnings_Date"] = pd.to_datetime(earnings_df["Earnings_Date"])
        except Exception:
            earnings_df = pd.DataFrame(columns=["Ticker", "Earnings_Date"])

        print(f"📋 Watchlist Sifted: {len(wl_df)} total -> {len(tickers)} enabled & passed score cutoff (>= {min_score})", flush=True)
        return tickers, sector_map, earnings_df

    def fetch_historical_bars(
        self,
        tickers: List[str],
        start_date: str = "2021-01-01",
        end_date: str = None,
        timeframe: TimeFrame = TimeFrame.Hour,
        force_refresh: bool = False,
        chunk_size: int = 50
    ) -> pd.DataFrame:
        """
        Fetches historical bars with incremental cache merging and real-time console updates.
        """
        cache_file = self.get_cache_file(timeframe)
        cached_df = pd.DataFrame()
        tickers_to_fetch = tickers

        # 1. Check existing disk cache
        if self.cache_exists(timeframe) and not force_refresh:
            try:
                cached_df = pd.read_pickle(cache_file)
                cached_tickers = set(cached_df.index.get_level_values("Ticker").unique())
                tickers_to_fetch = [t for t in tickers if t not in cached_tickers]

                if not tickers_to_fetch:
                    print(f"💾 [CACHE HIT] All {len(tickers)} requested tickers are already cached in '{cache_file.name}'.", flush=True)
                    return cached_df.loc[cached_df.index.get_level_values("Ticker").isin(tickers)]

                print(f"💾 [CACHE PARTIAL] Found {len(cached_tickers)} tickers in cache.", flush=True)
                print(f"🔄 Fetching delta: {len(tickers_to_fetch)} new/missing tickers...", flush=True)
            except Exception as e:
                print(f"⚠️ Cache read error: {e}. Refetching full dataset.", flush=True)
                tickers_to_fetch = tickers

        if not tickers_to_fetch:
            return cached_df

        if self.client is None:
            raise ValueError("Cannot fetch fresh data: Alpaca API credentials not initialized.")

        # 2. Fetch missing tickers from Alpaca
        start_dt = pd.to_datetime(start_date).tz_localize("America/New_York")
        end_dt = pd.to_datetime(end_date).tz_localize("America/New_York") if end_date else datetime.now(tz=start_dt.tz)

        new_bars = []
        total_chunks = (len(tickers_to_fetch) + chunk_size - 1) // chunk_size
        total_records_pulled = 0
        start_time = time.time()

        print(f"🌐 Fetching {len(tickers_to_fetch)} ticker bars ({start_date} to {end_date or 'Present'}) via IEX feed...", flush=True)

        for i in range(0, len(tickers_to_fetch), chunk_size):
            chunk = tickers_to_fetch[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1

            try:
                params = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                    feed=DataFeed.IEX
                )
                bars_response = self.client.get_stock_bars(params)
                if not bars_response.df.empty:
                    df_chunk = bars_response.df
                    new_bars.append(df_chunk)
                    total_records_pulled += len(df_chunk)
            except Exception as e:
                print(f"⚠️ Error fetching chunk {chunk_num} ({chunk[:3]}...): {e}", flush=True)

            # Clean scrolling progress bar output
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            pct = (chunk_num / total_chunks) * 100
            bar_len = 20
            filled = int(bar_len * chunk_num // total_chunks)
            bar = "#" * filled + "-" * (bar_len - filled)

            print(
                f"[{bar}] {pct:5.1f}% | Chunk {chunk_num}/{total_chunks} "
                f"({len(chunk)} tickers) | Records: +{total_records_pulled:,} | Time: {mins:02d}:{secs:02d}",
                flush=True
            )

        if not new_bars and cached_df.empty:
            raise RuntimeError("No bar data returned from Alpaca.")

        # 3. Format and Merge
        if new_bars:
            fresh_df = pd.concat(new_bars)
            fresh_df.index.names = ["Ticker", "Datetime"]
            fresh_df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
                "trade_count": "Trades", "vwap": "VWAP"
            }, inplace=True)
            fresh_df = fresh_df.tz_convert("America/New_York", level="Datetime")

            if not cached_df.empty and not force_refresh:
                combined_df = pd.concat([cached_df, fresh_df]).sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            else:
                combined_df = fresh_df

            # 4. Save to Disk
            combined_df.to_pickle(cache_file)
            print(f"💾 [CACHE UPDATED] Saved {len(combined_df):,} total records ({len(combined_df.index.get_level_values('Ticker').unique())} unique tickers) to '{cache_file.name}'.", flush=True)
            return combined_df.loc[combined_df.index.get_level_values("Ticker").isin(tickers)]

        return cached_df.loc[cached_df.index.get_level_values("Ticker").isin(tickers)]