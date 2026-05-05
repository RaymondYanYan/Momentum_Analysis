# Fin Tracker: Quantitative Mean-Reversion Strategy

## 1. Workflow Overview

The strategy follows a rigorous, multi-stage quantitative research pipeline designed to identify and exploit statistical anomalies in S&P 500 equities.

1.  **Universe Scanning:** We evaluate ~240 S&P 500 constituents to identify assets with inherent mean-reverting properties.
2.  **Signal Generation:** A composite engine combines Ornstein-Uhlenbeck Kalman Filters (OU-KF), Momentum Regime detection, and Volatility analysis.
3.  **Dynamic Optimization:** Instead of static parameters, we use **Rolling Window Optimization** to tailor Z-score thresholds to each ticker's recent "personality."
4.  **Non-Parametric Inference:** We validate results using Bootstrap Confidence Intervals and Permutation Tests to ensure edges are not due to statistical noise.
5.  **Live Monitoring:** A 24/7 automated system scans the market every 30 minutes, sending real-time Telegram alerts for high-probability entries and exits.

---

## 2. Methodologies & Core Engine

### The "Adaptive" Signal Engine
The core logic relies on three independent modules working in concert:

*   **Mean-Reversion (OU-KF Z-Score):** We use a 1-D Kalman Filter to track the "hidden" equilibrium price of a stock. When the price deviates significantly (Z-score > Threshold), we bet on a return to the mean.
*   **Momentum Regime Filter:** To avoid "catching falling knives," we classify the market into regimes (e.g., `STRONG_DOWN`, `EXHAUSTED_UP`). We disable long signals during strong downward trends.
*   **Dynamic Volatility Scaling:** The entry threshold is not fixed. It widens during high-volatility regimes (to avoid noise) and tightens during quiet periods (to capture subtle wiggles).

### Non-Parametric Statistical Validation
To prevent overfitting, we reject standard t-tests in favor of:
*   **Pairs Bootstrap:** Resampling individual trades to build empirical Confidence Intervals for Sharpe and Win Rate.
*   **Permutation Tests:** Shuffling signal labels to calculate a p-value for the null hypothesis: *"This strategy is no better than random chance."*
*   **Rank Correlation:** Using Spearman’s $\rho$ to verify that stronger signals consistently predict larger returns.

---

## 3. Assumptions

1.  **Stationarity of Regimes:** We assume that while prices are non-stationary, the *behavior* of volatility and momentum regimes is persistent enough to be modeled over a 3-year rolling window.
2.  **Liquidity:** The strategy assumes we can enter/exit positions at the closing price of the 30-minute candle with minimal slippage (capped at 5 bps in backtests).
3.  **Mean-Reversion:** The fundamental premise is that large deviations from a Kalman-filtered equilibrium are overreactions that will correct within a 1–5 day horizon.

---

## 4. Backtesting Framework

*   **Period:** Primary validation used a **10-year window (2016–2026)** to capture the 2018 correction, the 2020 Pandemic Crash, and the 2022 Bear Market.
*   **Costs:** Every trade deducts a $1.00 flat commission and 5 basis points of slippage.
*   **Risk Management:** Every position is protected by an **ATR Trailing Stop** (2.0x ATR), ensuring that a "value trap" doesn't result in catastrophic loss.
*   **Position Sizing:** Equal-dollar allocation (10% of capital per trade) to prevent any single ticker from dominating the portfolio's variance.

---

## 5. Results: The "All-Weather" Watchlist

After scanning 240 tickers and applying rolling optimization, we identified 8 robust candidates. Performance is based on the 10-year historical average with dynamic thresholds.

| Ticker | Sector | Opt Z | 10Y Sharpe | Win% | Max DD |
|--------|--------|-------|------------|------|--------|
| **CMCSA** | Comm Services | 2.5 | 0.80 | 84% | -0.22% |
| **DDOG** | Technology | 2.0 | 0.78 | 82% | -1.23% |
| **ES** | Utilities | 3.0 | 0.34 | 72% | -1.12% |
| **AVB** | Real Estate | 2.5 | 0.31 | 73% | -2.00% |
| **CHTR** | Comm Services | 2.0 | 0.42 | 61% | -5.31% |
| **LRCX** | Technology | 2.0 | 0.18 | 61% | -4.10% |
| **EQR** | Real Estate | 2.0 | 0.11 | 61% | -1.84% |
| **UDR** | Real Estate | 2.0 | 0.06 | 64% | -3.90% |

---

## 6. Possible Risks & Mitigations

### A. Regime Shift Risk
*   **Risk:** If the market enters a prolonged "Super-Cycle" trend (like the 2017 Tech Rally), mean-reversion signals will consistently fail as the "mean" moves away faster than the price can revert.
*   **Mitigation:** The **Momentum Regime Filter** disables signals during `STRONG_UP` or `STRONG_DOWN` phases to sidestep these periods.

### B. Parameter Decay
*   **Risk:** The optimal Z-score threshold (e.g., 2.5 for AVB) may drift over time as market microstructure changes.
*   **Mitigation:** The **Rolling Optimizer** is designed to be run quarterly. It re-calibrates the thresholds based on the most recent 3 years of data.

### C. Low-Frequency Noise
*   **Risk:** With only 20–50 trades per year per ticker, the strategy is susceptible to "lumpy" returns where a single bad month can skew the annual Sharpe.
*   **Mitigation:** We use an **8-ticker diversified watchlist** across different sectors (Tech, REITs, Utilities) to smooth out the equity curve.

### D. Execution Risk
*   **Risk:** In a flash-crash scenario, the ATR stop-loss might gap down, resulting in a fill much lower than the intended stop price.
*   **Mitigation:** We focus on large-cap, highly liquid S&P 500 stocks where gap-risk is statistically lower than in small-cap equities.
