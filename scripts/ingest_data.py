"""
===============================================================================
SCRIPT: BATCH ALPACA 1-HOUR HISTORICAL INGESTION RUNNER
LOCATION: scripts/ingest_data.py
===============================================================================
"""

import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_engine.data.loader import MarketDataLoader


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("🚀 ALPACA 1-HOUR HISTORICAL DATA INGESTION RUNNER")
    print("=" * 60)

    config = load_settings()
    loader = MarketDataLoader(config)

    # 1. Prompt for minimum Score filter
    yaml_score = config.get("filters", {}).get("min_ticker_score", 1)
    score_input = input(f"👉 Enter minimum Ticker Score [0-10] (Press Enter for default: {yaml_score}): ").strip()
    min_score = int(score_input) if score_input.isdigit() else yaml_score

    # 2. Check cache status directly from loader
    force_refresh = False
    if loader.cache_exists():
        cached_count = len(loader.get_cached_tickers())
        cache_name = loader.get_cache_file().name
        print(f"\n💾 Detected existing cache: '{cache_name}' ({cached_count} tickers stored)")
        mode_input = input("👉 Ingest Mode: [I]ncremental (append missing tickers only) or [F]ull wipe? (default: I): ").strip().lower()
        if mode_input in ["f", "full"]:
            force_refresh = True
            print("⚠️ FULL WIPE selected: Re-downloading entire universe from 2021-01-01...", flush=True)
        else:
            print("⚡ INCREMENTAL selected: Preserving existing data and scanning for delta...", flush=True)
    else:
        print(f"\n🌐 No existing cache found at '{loader.get_cache_file().name}'. Initializing full download...", flush=True)

    # 3. Filter watchlist and run fetch
    tickers, _, _ = loader.load_watchlist(min_score=min_score)

    if not tickers:
        print("❌ No tickers passed the enabled/score filter. Exiting.", flush=True)
        return

    df_bars = loader.fetch_historical_bars(
        tickers=tickers,
        start_date="2021-01-01",
        force_refresh=force_refresh,
        chunk_size=50
    )

    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete! Total records ready in cache: {len(df_bars):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()