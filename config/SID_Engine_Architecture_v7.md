# 🤖 SID Equities: Autonomous Trading Engine Architecture

**System Version:** v7.0 (Live Alpaca Integration)
**Strategy Mode:** Long-Only Equity Swing 
**Brokerage:** Alpaca (Paper / Live)

---

## 1. Core Operating Cadence
The engine operates autonomously on a **7-bar daily schedule** using 60-minute timeframe bars. It sleeps between scans and executes immediately at the top of the hour.
*   **Scan 1:** 10:30 AM 
*   **Scan 2:** 11:30 AM
*   **Scan 3:** 12:30 PM
*   **Scan 4:** 01:30 PM
*   **Scan 5:** 02:30 PM
*   **Scan 6:** 03:30 PM
*   **Scan 7 (Pre-Close):** 03:58 PM *(Calculates pro-rated volume based on 28 elapsed minutes)*

---

## 2. Entry Technicals & ML Gatekeeper
New entries require alignment between technical indicators and the machine learning model.

### Technical Setup Requirements
*   **Trend Filter:** Price must be above the 200 EMA.
*   **Momentum Hooks:** RSI must hook up within the last 3 bars, and MACD must hook up.
*   **Regime Identification (L_ID):** 
    *   *Aggressive Regime:* RSI drops below 30.
    *   *Tier/Standard Regime:* RSI drops below 35.

### Oracle v6.0 Hourly (The ML Gatekeeper)
If the technicals align, the engine builds a feature profile and queries the Oracle model. 
*   **Features Evaluated:** `RSI_Extreme_Value`, `Dist_From_EMA_%`, `Entry_RVOL`, `Entry_ATR`, `Days_To_Earnings`, `Days_Since_Earnings`, `Entry_Month`, `Risk_Tier`.
*   **The Veto (Trap Threshold):** If the model predicts a `P(Trap) >= 35%`, the trade is instantly vetoed and permanently blocked for that specific setup regime.
*   **The Whale (Whale Threshold):** If the model predicts a `P(Whale) >= 42%`, the system upgrades the exit strategy.

---

## 3. Position Sizing & Margin Management (The Cash Collar)
The engine strictly manages capital to prevent over-leveraging and margin calls.

*   **Sizing Base:** Standard risk base is $30,000.
*   **Risk Tiers:** 2% allocated for Aggressive setups; 0.5% allocated for Standard/Tier setups.
*   **Max Portfolio Exposure:** Engine is capped at a maximum of **15 open positions**.
*   **Max Single Asset Exposure:** No single position can exceed **20%** of total account equity.
*   **STRICT CASH COLLAR:** The engine physically queries Alpaca for `available_cash` before every purchase. Sizing is hard-capped to `min(risk_shares, max_notional_shares, max_cash_shares)`. **Zero margin is permitted.** If cash is insufficient to buy the setup outright, the trade is skipped.

---

## 4. Exit Strategy & Profit Taking
Exits are executed aggressively to lock in profits or mathematically define risk.

*   **Hard OTO Stop-Losses:** Every entry is sent as a `MarketOrder` combined with an `OrderClass.OTO` (One-Triggers-Other) Stop-Loss. The stop-loss is rigidly placed at the lowest low of the current setup regime (`SL_L`).
*   **Standard Take-Profit:** Executed immediately during an hourly scan if current RSI >= 50.
*   **Whale Take-Profit:** If the setup passed the Oracle's >42% Whale threshold, the take-profit target is expanded to RSI >= 60.

---

## 5. Defensive Systems & Risk Management
The engine features autonomous circuit breakers to protect capital during hostile market conditions.

*   **Drawdown Governor (Active):** The system tracks the portfolio's All-Time High Water Mark (HWM). If total portfolio equity drops **5%** from the HWM, the engine shifts into `🛡️ DEFENSIVE` mode. All sizing risk tiers are instantly slashed in half (max 1% for Aggressive, 0.25% for Tier).
*   **Regime Burn Guard (Active):** Once a trade is closed (either by profit-taking or stopping out), that specific `L_ID` regime is added to a local `burned_regimes` dictionary. The engine is permanently blocked from revenge-trading or "double-dipping" the same exact setup.
*   **Macro Blackout Calendar (`blackout_dates.json` — Planned / To Be Implemented):** Designed to check a master JSON file containing UTC timestamps of high-impact macroeconomic events (CPI, PPI, NFP, FOMC, BOJ, ECB). When implemented, if a scan occurs within the defined buffer window (e.g., 60 mins before to 120 mins after FOMC), **all new entries will be halted**. Open positions will continue to run and manage their stops/profits.
---

## 6. Infrastructure & Ledgering
*   **Stop-Loss Interceptor:** If a trade stops out between hourly scans, the engine detects the missing position in Alpaca, queries the closed order API, calculates the exact realized PnL/exit price, and securely logs it.
*   **Excel Live Ledger:** Every closed trade is appended to `SIM_Results/Alpaca_Live_Ledger.xlsx` to maintain a perfect accounting record outside of the Alpaca environment.
*   **Transient API Retry Decorator:** All Alpaca API calls (`get_account`, `get_stock_bars`, `submit_order`) are wrapped in a 3-attempt retry loop with a 5-second delay to safely navigate sudden `HTTP 500 Internal Server Errors` without crashing the daemon.

---

## 7. Oracle v6.0 Training Methodology (The Current Model)
The current ML Gatekeeper (Oracle v6.0 Hourly) was trained to act as a specialized classification engine to filter out false breakouts and "falling knife" traps in the 60-minute timeframe.

*   **Training Data Profile:** Built using historically backtested 60-minute bar data (7-bar daily cadence) across the SID Equities watchlist.
*   **Target Classification (The Labels):** 
    *   **Trap (Class 0):** The setup was triggered, but the price hit the lower-bound hard stop-loss (`SL_L`) *before* it could reach the RSI 50 profit target.
    *   **Standard Profit (Class 1):** The setup successfully triggered and reached the RSI 50 profit target without hitting the stop-loss.
    *   **Whale (Class 2):** Exceptional momentum setups that rapidly exceeded the RSI 60 target.
*   **Feature Engineering:** The model evaluates 8 distinct variables at the exact moment of technical alignment: `RSI_Extreme_Value`, `Dist_From_EMA_%`, `Entry_RVOL`, `Entry_ATR`, `Days_To_Earnings`, `Days_Since_Earnings`, `Entry_Month`, and `Risk_Tier`.
*   **Output:** Rather than a binary "Buy/No-Buy," it outputs an array of probabilities (`P_Trap`, `P_Standard`, `P_Whale`), allowing the engine to mathematically weigh the risk before deploying capital.

---

## 8. Oracle v7 Live Data Collection (The Feedback Loop)
To continuously improve the engine and adapt to changing macro-economic environments, `paper_trader.py` runs a background telemetry process via the `log_comprehensive_feature_snapshot()` function.

### What is Being Collected?
Every time the engine's technicals align (regardless of whether the trade is executed or vetoed), a permanent record is written to `data/training_store/oracle_v7_live_training_store.parquet`. 
The collected data includes:
*   **The Baseline Features:** The exact values of the 8 training features at the moment of the scan.
*   **Contextual Metadata:** `Ticker`, `Sector_Name`, `Watchlist_Score`, `Applied_Risk_Pct`, `Stop_Loss`, and `Entry_Price`.
*   **Model Confidence:** The exact `P(Trap)` and `P(Whale)` probabilities generated by Oracle v6.0.
*   **Engine Decision (`Action_Taken`):** Specifically logging whether the engine `BOUGHT` the stock or `VETOED` it.

### Why is this Collected for v7?
Historical backtesting is prone to hindsight bias. This live Parquet store represents pure, out-of-sample forward-testing. It will be the foundational dataset for training **Oracle v7** for three critical reasons:

1.  **Validating the Vetoes (The "Ghost" Ledger):** Because we log trades that the engine *refused* to take (`Action_Taken: VETOED`), we can later back-calculate the charts to see what *would* have happened. Did the veto save us from a crash, or did it block a massive rally? This allows v7 to reduce "False Positives" and dial in the perfect `trap_threshold`.
2.  **Sector-Specific Decay:** By logging the `Sector_Name`, v7 will be able to learn if certain features fail in specific industries (e.g., perhaps high RVOL is great for Tech, but a trap for Utilities).
3.  **Combating Concept Drift:** Market regimes change over time. Storing live feature snapshots allows the next iteration of the model to weight recent live market data more heavily than older historical data, keeping the ML Gatekeeper perfectly in sync with current volatility.

---

## 9. Autonomous vs. Discretionary Execution (The "Hands-Off" Edge)
Transitioning the SID Equities strategy from a manual, screen-heavy approach to a fully autonomous engine shifts the trading edge from human intuition to computational discipline. 

*   **Psychological Friction & Execution:** Manual trading is vulnerable to FOMO, hesitation, and rule-bending (e.g., taking a trade because the chart "looks good" despite a missing technical hook). The autonomous engine is mathematically ruthless. If the Oracle Gatekeeper flags a 35% trap probability, the trade is vetoed instantly without emotion.
*   **Time & Attention:** Manual swing trading requires monitoring the charts at the top of every hour, manually cross-referencing macro calendars, and calculating risk sizes on the fly. The autonomous bot operates on a precise 7-bar daily schedule, handling data ingestion, macro-checks, sizing, and execution in milliseconds, freeing the operator to focus purely on strategy architecture rather than babysitting positions.
*   **Absolute Risk Enforcement:** Human error (fat-finger mistakes, forgetting hard stop-losses, accidentally tapping into margin) is eliminated. The engine's "Cash Collar" physically audits the broker's cash balance before every execution, ensuring zero margin is used. The Drawdown Governor instantly halves risk sizing during a 5% portfolio dip—a level of defensive discipline that humans consistently struggle to enforce manually.
*   **Unbiased Data Collection:** Human trade journaling is historically biased, as traders rarely log the exact technical parameters of the trades they *chose not to take*. The engine's Parquet "Ghost Ledger" permanently logs every vetoed setup with its accompanying feature state, providing a pure, out-of-sample dataset to continuously refine and train future models (Oracle v7).

---

## 10. Environment, Deployment & Version Control
The logic is sound, but a future instance of this chat needs to know exactly *where* and *how* this engine lives. 
*   **Execution Environment:** The daemon relies on local terminal execution. When migrating from local IDE testing (like PyCharm) to a dedicated server or background process via Cygwin, the absolute paths in `PROJECT_ROOT` must remain stable. 
*   **Version Control Protocol:** Because the system dynamically writes to `alpaca_state.json`, `SIM_Results/Alpaca_Live_Ledger.xlsx`, and the Parquet training store, these specific data directories must be strictly added to `.gitignore`. Furthermore, when pushing updates via Git, careful routing of dual SSH identity keys is required to separate this proprietary trading repository from other organizational or work-related codebases. 
*   **API Security:** The engine strictly isolates credentials using a `.env` file mapped to `os.environ`. These keys are never hardcoded.

---

## 11. The Forex Expansion Roadmap (The 24/5 Problem)
The architecture is currently hardcoded for the "SID Equities" 7-bar daily cadence (NY market hours). Moving into currency pairs will break this time-handling logic. Future development must address:
*   **Timeframe Overhaul:** Forex trades 24/5. The engine will need a secondary loop or a completely separate `forex_trader.py` daemon that does not rely on the 9:30 AM to 4:00 PM EST market clock. 
*   **Framework Integration:** Future iterations will need to map these ML gatekeeping and risk-collar concepts into your existing `SimonPullen_FX` or `DeniDantev_FX` trading frameworks. 
*   **Asset-Specific Blackouts:** The JSON macro calendar is already configured to isolate blackouts by currency (e.g., halting EUR pairs during an ECB rate decision while letting USD/JPY run). The Forex execution script must be built to parse the `"currency": ["EUR"]` arrays in the JSON, rather than applying a blanket halt to the entire engine.

---

## 12. Known System Vulnerabilities & Data Fallbacks
A future engineer (or AI) needs to know the system's current Achilles' heels to avoid debugging wild goose chases.
*   **The IEX Data Dependency:** The current `DataFeed.IEX` is free but prone to occasional volume dropouts. If the feed returns an empty dataframe, the engine currently skips the hour. Future upgrades should include a data-fallback protocol (e.g., querying Yahoo Finance or Polygon.io if Alpaca's IEX feed fails).
*   **The Oracle v7 Retraining Trigger:** The Parquet training store is infinitely collecting data, but there is no defined threshold for *when* to train v7. Future development must define a chronological or sample-size trigger (e.g., retrain every 90 days or after 1,000 new logged setups) to prevent the ML model from becoming stale.

---

## 13. AI Co-Pilot Persona & Engineering Directives

When initializing an AI assistant or LLM session with this architectural blueprint, the model must adopt the following persona, expertise profile, and operational constraints:

### Role & Domain Expertise
* **Role:** Lead Quantitative Systems Engineer & Systematic Multi-Asset Trader (Equities & Spot FX).
* **Primary Responsibilities:** 
  * Architect, audit, and refactor clean, production-grade Python code tailored for automated broker APIs (Alpaca, ccxt, Interactive Brokers).
  * Enforce strict algorithmic risk boundaries, money management protocols, and statistical discipline.
  * Guide feature engineering, training pipelines, and evaluation metrics for machine learning classification engines (Oracle series).
  * Design fault-tolerant execution loops, state synchronizers, and macro event filters.

---

### Core Engineering Directives
1. **Rule-Based Rigor (No Discretionary Drift):** Treat all entry, exit, regime identification, and sizing criteria as immutable programmatic laws. Never suggest discretionary "eyeball" exceptions or emotional overrides.
2. **Defensive Capital First:** Prioritize margin avoidance (the Strict Cash Collar), hard stop-loss placement, and drawdown governance over aggressive leverage or trade frequency.
3. **Execution & Time Mechanics:**
   * Respect bar-closing cadence vs. real-time execution nuances (e.g., handling 60m closed bars vs. the 28-minute pro-rated pre-close bar).
   * Maintain time-zone consistency across UTC event timestamps, NY market clocks (`America/New_York`), and 24/5 continuous Forex sessions.
4. **State & Environment Hygiene:**
   * Keep state persistence (`alpaca_state.json`, Parquet feature stores, Excel ledgers) decoupled and protected via `.gitignore`.
   * Ensure API credentials remain isolated in `.env` and all external I/O operations include retry wrappers for transient network or broker errors.