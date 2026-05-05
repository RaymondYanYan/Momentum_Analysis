"""
Signal Monitor: Continuous watchlist scanner with Telegram alerts.

Monitors the 8-ticker PRIORITIZE watchlist for new mean-reversion signals
and sends real-time Telegram alerts on entry/exit events.

Architecture
------------
1. **Signal Scanner**: Runs `signal_engine.generate_signals()` on watchlist
2. **State Tracker**: Remembers open positions to detect new entries/exits
3. **Alert Dispatcher**: Sends formatted alerts via Telegram Bot
4. **Scheduler**: Runs every 30 min during market hours (9:30-16:00 ET)

State Persistence
-----------------
Open positions are tracked in-memory during a session. For production use,
persist state to disk (JSON) so restarts don't lose position tracking.

Usage
-----
    # Set credentials
    export TELEGRAM_BOT_TOKEN="..."
    export TELEGRAM_CHAT_ID="..."

    # Run monitor (continuous, 30-min intervals)
    ./venv/bin/python monitor.py

    # Run once (single scan, for testing)
    ./venv/bin/python monitor.py --once

    # Custom interval
    ./venv/bin/python monitor.py --interval 15
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from core.price_collector import PriceDataCollector
from core.signal_engine import generate_signals
from alerts.telegram_alerts import TelegramBot, WATCHLIST, TICKER_META, SECTOR_CAPS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_FILE = Path(__file__).parent / "monitor_state.json"
MARKET_OPEN_ET = 9 * 60 + 30  # 9:30 AM ET in minutes
MARKET_CLOSE_ET = 16 * 60  # 4:00 PM ET in minutes
ET_UTC_OFFSET = -5  # ET is UTC-5 (adjust for DST if needed; simplification)

# Signal parameters (match backtest config)
ZSCORE_THRESHOLD = 2.0
RSI_OVERSOLD = 25.0
RSI_OVERBOUGHT = 75.0
ATR_STOP_MULTIPLIER = 2.0
POSITION_SIZE_PCT = 0.10
INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Position state management
# ---------------------------------------------------------------------------

class PositionTracker:
    """Tracks open positions and detects new entries/exits.

    Persists state to disk so restarts don't lose position data.
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.positions: Dict[str, dict] = self._load_state()

    def _load_state(self) -> Dict[str, dict]:
        """Load positions from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                logger.info("Loaded %d open positions from %s", len(state), self.state_file)
                return state
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load state file: %s", e)
        return {}

    def _save_state(self):
        """Save positions to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.positions, f, indent=2)
        except IOError as e:
            logger.error("Failed to save state file: %s", e)

    def get_open_position(self, ticker: str) -> Optional[dict]:
        """Get open position for a ticker, or None."""
        return self.positions.get(ticker)

    def open_position(
        self,
        ticker: str,
        signal_type: str,
        entry_price: float,
        zscore: float,
        rsi: float,
        atr_stop: float,
        atr_value: float,
    ):
        """Record a new open position."""
        self.positions[ticker] = {
            "ticker": ticker,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "zscore": zscore,
            "rsi": rsi,
            "atr_stop": atr_stop,
            "atr_value": atr_value,
            "status": "open",
        }
        self._save_state()
        logger.info("Opened position: %s %s @ $%.2f", ticker, signal_type, entry_price)

    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str,
        pnl_pct: float,
        pnl_dollar: float,
    ) -> Optional[dict]:
        """Close a position and return its data for the exit alert."""
        if ticker not in self.positions:
            return None

        position = self.positions.pop(ticker)
        position["exit_price"] = exit_price
        position["exit_time"] = datetime.now(timezone.utc).isoformat()
        position["exit_reason"] = exit_reason
        position["pnl_pct"] = pnl_pct
        position["pnl_dollar"] = pnl_dollar
        position["status"] = "closed"

        self._save_state()
        logger.info(
            "Closed position: %s | Reason: %s | PnL: $%.2f (%.2f%%)",
            ticker, exit_reason, pnl_dollar, pnl_pct,
        )
        return position

    def get_all_open_positions(self) -> List[dict]:
        """Return all currently open positions."""
        return list(self.positions.values())

    def check_exit_conditions(
        self,
        ticker: str,
        current_signal: str,
        current_price: float,
        atr_14: float,
        zscore: float = 0.0,  # Added for "Exit at Mean" logic
    ) -> Optional[dict]:
        """Check if an open position should be exited.

        Returns position data if exit triggered, None otherwise.
        """
        position = self.get_open_position(ticker)
        if position is None:
            return None

        entry_price = position["entry_price"]
        signal_type = position["signal_type"]
        atr_stop = position["atr_stop"]

        exit_reason = None

        # 1. Signal reversal or Return to Mean
        if "LONG" in signal_type:
            if current_signal in ("SHORT_REVERSION", "SHORT_BOUNCE", "NEUTRAL"):
                exit_reason = "SIGNAL_REVERSAL"
            elif abs(zscore) < 0.5:  # Exit at Mean
                exit_reason = "RETURN_TO_MEAN"
            elif current_price < atr_stop:
                exit_reason = "ATR_STOP_LOSS"
        elif "SHORT" in signal_type:
            if current_signal in ("LONG_REVERSION", "LONG_BOUNCE", "NEUTRAL"):
                exit_reason = "SIGNAL_REVERSAL"
            elif abs(zscore) < 0.5:  # Exit at Mean
                exit_reason = "RETURN_TO_MEAN"
            elif current_price > atr_stop:
                exit_reason = "ATR_STOP_LOSS"

        if exit_reason:
            # Calculate PnL
            if "LONG" in signal_type:
                pnl_pct = (current_price / entry_price) - 1.0
            else:
                pnl_pct = 1.0 - (current_price / entry_price)

            pnl_dollar = pnl_pct * INITIAL_CAPITAL * POSITION_SIZE_PCT

            return self.close_position(ticker, current_price, exit_reason, pnl_pct, pnl_dollar)

        return None


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30 AM - 4:00 PM ET).

    Simplified: does not account for holidays or DST transitions.
    """
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc + timedelta(hours=ET_UTC_OFFSET)
    current_minutes = now_et.hour * 60 + now_et.minute

    return MARKET_OPEN_ET <= current_minutes <= MARKET_CLOSE_ET and now_et.weekday() < 5


# ---------------------------------------------------------------------------
# Signal scanner
# ---------------------------------------------------------------------------

class SignalScanner:
    """Scans the watchlist for new signals and triggers alerts."""

    def __init__(self, bot: TelegramBot, tracker: PositionTracker):
        self.bot = bot
        self.tracker = tracker

    def scan_once(self, tickers: List[str] = None) -> dict:
        """Run a single scan on the watchlist.

        Parameters
        ----------
        tickers : list of str
            Tickers to scan. Defaults to WATCHLIST.

        Returns
        -------
        dict with stats: new_signals, exits, errors
        """
        if tickers is None:
            tickers = WATCHLIST

        stats = {"new_signals": 0, "exits": 0, "errors": 0}

        logger.info("Scanning %d tickers: %s", len(tickers), ", ".join(tickers))

        try:
            # Process each ticker individually to apply ticker-specific thresholds
            all_signals = []
            for ticker in tickers:
                try:
                    collector = PriceDataCollector(tickers=[ticker], start="2023-01-01")
                    prices = collector.collect()
                    
                    # Use ticker-specific threshold from RESI-4 metadata
                    thresh = TICKER_META[ticker]["zscore_threshold"]
                    
                    sig = generate_signals(
                        prices,
                        zscore_threshold=thresh,
                        rsi_oversold=RSI_OVERSOLD,
                        rsi_overbought=RSI_OVERBOUGHT,
                    )
                    all_signals.append(sig)
                except Exception as e:
                    logger.error("Failed to process %s: %s", ticker, e)
                    stats["errors"] += 1

            if not all_signals:
                return stats
                
            combined_signals = pd.concat(all_signals, ignore_index=True)

            # Process each ticker
            for ticker in tickers:
                try:
                    ticker_signals = combined_signals[combined_signals["ticker"] == ticker]
                    if ticker_signals.empty:
                        logger.warning("No signal data for %s", ticker)
                        continue

                    # Get latest row
                    latest = ticker_signals.iloc[-1]
                    current_signal = latest["signal"]
                    current_price = latest["close"]
                    zscore = latest.get("ou_kf_zscore", 0.0)
                    rsi = latest.get("rsi_14", 50.0)
                    atr_14 = latest.get("atr_14", 0.0)
                    atr_stop = current_price - ATR_STOP_MULTIPLIER * atr_14 if "LONG" in str(current_signal) else current_price + ATR_STOP_MULTIPLIER * atr_14

                    # Check for exit first (if position is open)
                    exit_data = self.tracker.check_exit_conditions(
                        ticker, current_signal, current_price, atr_14, zscore
                    )
                    if exit_data:
                        self.bot.send_exit_alert(
                            ticker=ticker,
                            exit_reason=exit_data["exit_reason"],
                            entry_price=exit_data["entry_price"],
                            exit_price=exit_data["exit_price"],
                            pnl_pct=exit_data["pnl_pct"] * 100,
                            pnl_dollar=exit_data["pnl_dollar"],
                        )
                        stats["exits"] += 1
                        continue  # Don't check for new entry if we just exited

                    # Check for new entry (if no position is open)
                    if self.tracker.get_open_position(ticker) is None:
                        if current_signal in ("LONG_REVERSION", "SHORT_REVERSION", "LONG_BOUNCE", "SHORT_BOUNCE"):
                            # --- Sector Capping Logic ---
                            sector = TICKER_META[ticker]["sector"]
                            sector_exposure = sum(1 for p in self.tracker.get_all_open_positions() 
                                                  if TICKER_META.get(p["ticker"], {}).get("sector") == sector)
                            
                            if sector_exposure >= SECTOR_CAPS.get(sector, 99):
                                logger.info("Sector cap reached for %s. Skipping %s.", sector, ticker)
                                continue

                            # Determine ATR stop for the new position
                            if "LONG" in current_signal:
                                atr_stop = current_price - ATR_STOP_MULTIPLIER * atr_14
                            else:
                                atr_stop = current_price + ATR_STOP_MULTIPLIER * atr_14

                            self.tracker.open_position(
                                ticker=ticker,
                                signal_type=current_signal,
                                entry_price=current_price,
                                zscore=zscore,
                                rsi=rsi,
                                atr_stop=atr_stop,
                                atr_value=atr_14,
                            )

                            self.bot.send_signal_alert(
                                ticker=ticker,
                                signal_type=current_signal,
                                price=current_price,
                                zscore=zscore,
                                rsi=rsi,
                                atr_stop=atr_stop,
                                atr_value=atr_14,
                            )
                            stats["new_signals"] += 1

                except Exception as e:
                    logger.error("Error processing %s: %s", ticker, e)
                    stats["errors"] += 1

        except Exception as e:
            logger.error("Scan failed: %s", e)
            stats["errors"] += 1

        logger.info(
            "Scan complete: %d new signals, %d exits, %d errors",
            stats["new_signals"], stats["exits"], stats["errors"],
        )
        return stats

    def send_daily_summary(self):
        """Send end-of-day summary of all open positions with enhanced information."""
        positions = self.tracker.get_all_open_positions()

        # Calculate daily PnL (sum of all open position PnLs)
        daily_pnl = sum(p.get("pnl_dollar", 0) for p in positions)

        # Add current prices and enhanced data to position data
        enriched_positions = []
        for pos in positions:
            enriched_positions.append({
                "ticker": pos["ticker"],
                "signal_type": pos["signal_type"],
                "entry_price": pos["entry_price"],
                "current_price": pos["entry_price"],  # Placeholder; fetch real current price if needed
                "pnl_pct": pos.get("pnl_pct", 0) * 100 if pos.get("pnl_pct") else 0,
                "pnl_dollar": pos.get("pnl_dollar", 0),
                "zscore": pos.get("zscore", 0),
                "rsi": pos.get("rsi", 50),
            })

        # Calculate sector allocation
        sector_allocation = {}
        for pos in positions:
            ticker = pos["ticker"]
            sector = TICKER_META.get(ticker, {}).get("sector", "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0) + 1

        # Placeholder for watchlist performance (would need historical data)
        watchlist_performance = {
            "win_rate": 0,  # Would be calculated from trade history
            "avg_return": 0,  # Would be calculated from trade history
            "total_trades": 0,  # Would be from trade history
        }

        # Placeholder for risk metrics (would need more complex calculations)
        risk_metrics = {
            "max_drawdown": 0,  # Would be calculated from equity curve
            "sharpe_estimate": 0,  # Would be calculated from returns
            "volatility_percentile": 0,  # Would be calculated from volatility data
        }

        # Placeholder for market context (would need external data)
        market_context = {
            "regime": "UNKNOWN",  # Would come from regime detection
            "volatility_regime": "UNKNOWN",  # Would come from volatility analysis
        }

        self.bot.send_daily_summary(
            positions=enriched_positions,
            daily_pnl=daily_pnl,
            active_signals=len(positions),
            watchlist_performance=watchlist_performance,
            sector_allocation=sector_allocation,
            risk_metrics=risk_metrics,
            market_context=market_context,
        )


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_monitor(interval_minutes: int = 30, run_once: bool = False):
    """Run the signal monitor.

    Parameters
    ----------
    interval_minutes : int
        Seconds between scans.
    run_once : bool
        If True, run a single scan and exit (for testing).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).parent / "monitor.log"),
        ],
    )

    # Initialize components
    try:
        bot = TelegramBot()
    except ValueError as e:
        logger.error("Failed to initialize Telegram bot: %s", e)
        logger.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        sys.exit(1)

    tracker = PositionTracker()
    scanner = SignalScanner(bot, tracker)

    logger.info("=" * 60)
    logger.info("Signal Monitor started")
    logger.info("Watchlist: %s", ", ".join(WATCHLIST))
    logger.info("Check interval: %d minutes", interval_minutes)
    logger.info("=" * 60)

    if run_once:
        scanner.scan_once()
        return

    # Main loop
    while True:
        if is_market_hours():
            logger.info("Market is open. Running scan...")
            scanner.scan_once()
        else:
            logger.info("Market is closed. Skipping scan.")

        # Wait until next interval
        next_check = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
        logger.info("Next check at %s UTC", next_check.strftime("%H:%M"))
        time.sleep(interval_minutes * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Signal Monitor with Telegram Alerts")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in minutes (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    args = parser.parse_args()

    run_monitor(interval_minutes=args.interval, run_once=args.once)
