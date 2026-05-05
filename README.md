# Fin Tracker — Mean-Reversion Signal System

A quantitative trading system that detects mean-reversion opportunities in S&P 500 stocks using Ornstein-Uhlenbeck Kalman Filter Z-scores, momentum regime filters, and ATR-based risk management.

## Architecture

```
Fin_tracker/
├── core/                    # Core engine
│   ├── price_collector.py   # yfinance data downloader
│   ├── mean_reversion.py    # Rolling Z-score + OU-KF Z-score
│   ├── momentum.py          # ROC, RSI, regime classification
│   ├── volatility.py        # ATR, volatility percentile
│   ├── signal_engine.py     # Unified signal aggregator
│   ├── backtest.py          # Vectorized backtesting engine
│   ├── statistical_inference.py  # Bootstrap, permutation tests, KDE
│   └── tuning.py            # Grid-search parameter optimization
├── alerts/                  # Telegram alert system
│   ├── telegram_alerts.py   # Telegram Bot API wrapper
│   └── monitor.py           # Continuous watchlist scanner
├── data/                    # Scan results and datasets
│   ├── universe_scan.py     # S&P 500 universe scanner
│   └── sp500_scan_results.csv
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests (placeholder)
├── .env                     # Telegram credentials (gitignored)
├── run_monitor.sh           # Entry point for live monitoring
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd Fin_tracker
./venv/bin/pip install -r requirements.txt  # if you create one
# Or manually:
./venv/bin/pip install yfinance pandas numpy scipy scikit-learn python-dotenv requests
```

### 2. Configure Telegram Alerts



**To get your Chat ID:**
1. Message your bot on Telegram (search by its username)
2. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Look for `"chat":{"id":123456789}` in the response

### 3. Test the Bot

```bash
./venv/bin/python alerts/telegram_alerts.py --test
```

### 4. Run a Backtest

```bash
# Single ticker
./venv/bin/python core/backtest.py

# With non-parametric statistics
./venv/bin/python core/backtest.py --stats --ticker VOO --start 2020-01-01

# Custom parameters
./venv/bin/python core/backtest.py --stats --ticker AAPL --zscore 1.5 --rsi-oversold 30
```

### 5. Scan the S&P 500 Universe

```bash
./venv/bin/python data/universe_scan.py --start 2020-01-01
```

Results are saved to `data/sp500_scan_results.csv`.

### 6. Run the Live Monitor

```bash
# Single scan (testing)
./run_monitor.sh --once

# Continuous monitoring (30-min intervals during market hours)
./run_monitor.sh

# Custom interval (e.g., 15 min)
./run_monitor.sh --interval 15
```

## Signal Logic

### Mean Reversion (Primary)
- **OU-KF Z-score**: Kalman Filter tracking the equilibrium price of an Ornstein-Uhlenbeck process
  - `LONG` when Z < -2.0 and rolling Z(20) < -1.0
  - `SHORT` when Z > +2.0 and rolling Z(20) > +1.0

### Momentum Filter (Confirmation)
- **Regime classification**: STRONG_UP, STRONG_DOWN, EXHAUSTED_UP, EXHAUSTED_DOWN, NEUTRAL
- **ROC filter**: Don't catch falling knives (no LONG if 20-day ROC < -10%)
- **RSI thresholds**: Dynamic oversold/overbought levels

### Volatility (Risk Management)
- **ATR trailing stop**: Exit when price moves 2× ATR against the position
- **Volatility percentile**: Adjust Z-score thresholds in high-vol regimes

### Signal Types
| Signal | Condition |
|--------|-----------|
| `LONG_REVERSION` | OU-KF Z < -2.0, regime not STRONG_DOWN, ROC > -10% |
| `SHORT_REVERSION` | OU-KF Z > +2.0, regime not STRONG_UP, ROC < +10% |
| `LONG_BOUNCE` | EXHAUSTED_DOWN regime + RSI < 25 |
| `SHORT_BOUNCE` | EXHAUSTED_UP regime + RSI > 75 |

## Watchlist (PRIORITIZE)

Based on S&P 500 universe scan (2020-2026), these 8 tickers show the strongest mean-reversion behavior:

| Ticker | Sector | Sharpe | Win Rate |
|--------|--------|--------|----------|
| ESS | Real Estate | 0.86 | 82% |
| AVB | Real Estate | 0.71 | 86% |
| HST | Real Estate | 1.11 | 83% |
| PPL | Utilities | 0.99 | 76% |
| ES | Utilities | 0.85 | 76% |
| PG | Consumer Staples | 0.78 | 73% |
| ABT | Healthcare | 0.88 | 84% |
| BDX | Healthcare | 0.85 | 86% |

**Excluded**: SHOP, DDOG, ABNB, NKE, SBUX, CMCSA, ACN (momentum risk, structural decline, or excess noise).

## Non-Parametric Statistical Inference

The system validates signal edge using distribution-free methods:

1. **Bootstrap Confidence Intervals**: Resample trades to get empirical CIs on Sharpe, win rate, profit factor
2. **Permutation Tests**: Shuffle signal labels to test if observed performance could arise by chance
3. **Rank Correlation**: Spearman/Kendall tests for signal strength → return predictability
4. **Kernel Density Estimation**: Compare return distributions across signal types (KS test)

Run with:
```bash
./venv/bin/python core/backtest.py --stats --ticker ESS --start 2020-01-01
```

## Telegram Alert Format

### New Signal Alert
```
🟢 NEW SIGNAL: ESS
━━━━━━━━━━━━━━━━━━━━━━
📊 Sector: Real Estate
📈 Direction: LONG (Reversion)
💰 Entry Price: $52.34
📐 Z-Score: -2.31
📉 RSI(14): 23.4
🛑 ATR Stop: $50.12 (ATR=1.11)
⏰ Time: 2026-04-03 14:30 UTC
━━━━━━━━━━━━━━━━━━━━━━
```

### Exit Alert
```
✅ POSITION CLOSED: ESS
━━━━━━━━━━━━━━━━━━━━━━
🚪 Exit: Signal Reversal
📥 Entry: $52.34
📤 Exit: $53.10
📊 PnL: +1.45% ($145.20)
⏰ Time: 2026-04-03 15:00 UTC
━━━━━━━━━━━━━━━━━━━━━━
```

### Daily Summary
```
📋 DAILY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━
📅 Date: 2026-04-03
📊 Active Positions: 3
💰 Daily PnL: $234.50 ✅
━━━━━━━━━━━━━━━━━━━━━━
Open Positions:
  • ESS (LONG_REVERSION)
    Entry: $52.34 → Current: $52.80
    PnL: +0.88% ($88.00)
  • AVB (LONG_BOUNCE)
    Entry: $118.50 → Current: $119.20
    PnL: +0.59% ($59.00)
━━━━━━━━━━━━━━━━━━━━━━
```

## Configuration

Key parameters in `core/backtest.py` and `alerts/monitor.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zscore_threshold` | 2.0 | OU-KF Z-score trigger level |
| `rsi_oversold` | 25.0 | RSI level for bounce signals |
| `rsi_overbought` | 75.0 | RSI level for short bounce |
| `atr_stop_multiplier` | 2.0 | ATR multiples for stop loss |
| `position_size_pct` | 0.10 | Capital fraction per trade |
| `commission_per_trade` | 1.0 | Flat fee per trade |
| `slippage_bps` | 5.0 | Slippage in basis points |

## Future Work

- [ ] Walk-forward validation (currently has import issues)
- [ ] Persistence of monitor state across restarts (JSON state file exists but needs testing)
- [ ] Multi-ticker portfolio-level backtesting
- [ ] Position sizing optimization (Kelly criterion, volatility targeting)
- [ ] Regime detection for market-wide risk-off periods
- [ ] Unit tests for core modules

## License

Private use only.
