content = """## Data Feed & Volume Distortion
* **IEX Feed Limitations**: Requests utilize `DataFeed.IEX`, capturing only ~2% to 3% of total US equity volume.
* **Feature Skew**: Metrics like `Entry_RVOL` and `ATR` calculated on a restricted liquidity tape distort the ML Oracle's feature vector, inducing false confidence or missed signals.
* **Remediation**: Migrate the historical data client to the Consolidated Tape (`DataFeed.SIP`) to capture complete market depth and accurate volume figures.

## State Desynchronization & Crash Resilience
* **Local vs. Broker Drift**: State management relies entirely on `alpaca_state.json`. If an order fills on Alpaca's server but the script crashes before writing to disk, local memory desynchronizes from actual broker state.
* **Manual Override Blind Spots**: Manual closures via the Alpaca web dashboard create a mismatch between `trading_client.get_all_positions()` and `position_meta`, causing orphaned records or broken regime burn guards.
* **Remediation**: Implement defensive startup routines that query live broker state and reconcile or overwrite local JSON metadata automatically.

## Execution & Slippage Vulnerabilities
* **Market Order Exposure**: Entries and take-profits execute via `MarketOrderRequest`. Crossing the spread with market orders during volatile hourly closes or macro news shocks incurs severe slippage.
* **Stop-Loss Gap Risk**: While OTO stop-losses protect against runaway drawdowns, sharp inter-hour gaps convert stops into market orders, filling significantly worse than the designated `SL_L` floor.
* **Remediation**: Evaluate limit-buffer mechanics or volatility-adjusted execution wrappers to mitigate open-market slippage.

## API Payload & Rate Limiting Bottlenecks
* **Watchlist Scale**: The sifted watchlist tracks **445 enabled tickers**. Requesting hourly bars for all 445 symbols in a single monolithic `StockBarsRequest` risks payload limits, timeouts, or HTTP 429 rate-limiting errors.
* **Remediation**: Chunk the ticker array into batch requests of 50 to 100 symbols to guarantee reliable data ingestion without dropping symbols mid-scan.

## Clock Drift & Timing Edge Cases
* **Strict Second-Level Triggers**: Sleeping until `11:30:05` assumes absolute clock synchronization between the local machine and Alpaca servers. A minor local drift queries the API before the exchange fully indexes the closed bar, triggering empty DataFrames or indicator calculation errors.
* **Remediation**: Integrate API retry loops with backoff logic and bar-timestamp validation checks to verify that fully closed hourly candles are processed."""

with open("Vulnerabilities.md", "w", encoding="utf-8") as f:
    f.write(content)
print("Successfully generated Vulnerabilities.md in project root.")