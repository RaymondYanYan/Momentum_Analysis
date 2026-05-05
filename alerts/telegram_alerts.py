"""
Telegram Alert Bot for Mean-Reversion Signals.

Sends real-time trading alerts to a Telegram chat when the signal engine
detects entry/exit opportunities on the watchlist.

Features
--------
- New signal alerts (LONG/SHORT entries)
- Exit alerts (signal reversal or ATR stop hit)
- Daily summary of all active positions
- Error handling and rate limiting

Usage
-----
    ./venv/bin/python telegram_alerts.py --token TOKEN --chat-id CHAT_ID

Or set environment variables:
    export TELEGRAM_BOT_TOKEN="..."
    export TELEGRAM_CHAT_ID="..."
    ./venv/bin/python telegram_alerts.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Watchlist: The "RESI-4" High-Conviction REIT Bucket
WATCHLIST = ["AVB", "EQR", "INVH", "MAA"]

# Sector Capping: Max concurrent positions per sector
SECTOR_CAPS = {"REIT": 2}

# Ticker-specific metadata for risk management
TICKER_META = {
    "AVB": {"sector": "REIT", "zscore_threshold": 2.5},
    "EQR": {"sector": "REIT", "zscore_threshold": 2.5},
    "INVH": {"sector": "REIT", "zscore_threshold": 2.5},
    "MAA": {"sector": "REIT", "zscore_threshold": 2.5}
}

# Sector mapping for alert context
SECTOR_MAP = {
    "ESS": "Real Estate",
    "AVB": "Real Estate",
    "HST": "Real Estate",
    "PPL": "Utilities",
    "ES": "Utilities",
    "PG": "Consumer Staples",
    "ABT": "Healthcare",
    "BDX": "Healthcare",
}

# ---------------------------------------------------------------------------
# Telegram API wrapper
# ---------------------------------------------------------------------------


class TelegramBot:
    """Simple Telegram Bot wrapper using the HTTP API."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or BOT_TOKEN
        self.chat_id = chat_id or CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        if not self.token:
            raise ValueError("Telegram bot token not set. Set TELEGRAM_BOT_TOKEN env var or pass to constructor.")
        if not self.chat_id:
            raise ValueError("Telegram chat ID not set. Set TELEGRAM_CHAT_ID env var or pass to constructor.")

    def send_message(self, text: str, parse_mode: str = "HTML") -> dict:
        """Send a text message to the configured chat.

        Parameters
        ----------
        text : str
            Message text (supports HTML formatting if parse_mode='HTML')
        parse_mode : str
            'HTML' or 'MarkdownV2'

        Returns
        -------
        dict : Telegram API response
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("ok"):
                logger.info("Message sent successfully (msg_id=%s)", result["result"]["message_id"])
            else:
                logger.error("Telegram API error: %s", result.get("description", "Unknown"))

            return result

        except requests.exceptions.RequestException as e:
            logger.error("Failed to send Telegram message: %s", e)
            return {"ok": False, "error": str(e)}

    def send_signal_alert(
        self,
        ticker: str,
        signal_type: str,
        price: float,
        zscore: float,
        rsi: float,
        atr_stop: float,
        atr_value: float,
        signal_strength: str = "",
    ):
        """Send a formatted signal alert.

        Parameters
        ----------
        ticker : str
        signal_type : str
            'LONG_REVERSION', 'SHORT_REVERSION', 'LONG_BOUNCE', 'SHORT_BOUNCE'
        price : float
            Current / entry price
        zscore : float
            OU-KF Z-score
        rsi : float
            RSI(14)
        atr_stop : float
            ATR-based stop loss level
        atr_value : float
            Current ATR(14) value
        signal_strength : str
            Optional: 'STRONG', 'MODERATE', etc.
        """
        sector = SECTOR_MAP.get(ticker, "Unknown")

        # Emoji based on signal type
        if "LONG" in signal_type:
            emoji = "🟢"
            direction = "LONG"
        else:
            emoji = "🔴"
            direction = "SHORT"

        if "BOUNCE" in signal_type:
            subtype = "Bounce"
        else:
            subtype = "Reversion"

        # Format signal strength
        strength_badge = f" | <b>{signal_strength}</b>" if signal_strength else ""

        message = (
            f"{emoji} <b>NEW SIGNAL: {ticker}</b>{strength_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Sector: {sector}\n"
            f"📈 Direction: <b>{direction} ({subtype})</b>\n"
            f"💰 Entry Price: <b>${price:.2f}</b>\n"
            f"📐 Z-Score: {zscore:+.2f}\n"
            f"📉 RSI(14): {rsi:.1f}\n"
            f"🛑 ATR Stop: ${atr_stop:.2f} (ATR={atr_value:.2f})\n"
            f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )

        return self.send_message(message)

    def send_exit_alert(
        self,
        ticker: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_dollar: float,
        hold_duration: str = "",
    ):
        """Send a formatted exit/close alert.

        Parameters
        ----------
        ticker : str
        exit_reason : str
            'SIGNAL_REVERSAL', 'ATR_STOP_LOSS', 'END_OF_DATA'
        entry_price : float
        exit_price : float
        pnl_pct : float
            PnL as percentage
        pnl_dollar : float
        hold_duration : str
            Optional: '2 days', '5 days', etc.
        """
        pnl_emoji = "✅" if pnl_dollar > 0 else "❌"
        reason_label = {
            "SIGNAL_REVERSAL": "Signal Reversal",
            "ATR_STOP_LOSS": "ATR Stop Loss",
            "END_OF_DATA": "End of Day",
        }.get(exit_reason, exit_reason)

        duration_str = f"\n⏱ Hold: {hold_duration}" if hold_duration else ""

        message = (
            f"{pnl_emoji} <b>POSITION CLOSED: {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🚪 Exit: {reason_label}\n"
            f"📥 Entry: ${entry_price:.2f}\n"
            f"📤 Exit: ${exit_price:.2f}\n"
            f"📊 PnL: <b>{pnl_pct:+.2f}%</b> (${pnl_dollar:+.2f}){duration_str}\n"
            f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )

        return self.send_message(message)

    def send_daily_summary(self, positions: list[dict], daily_pnl: float, active_signals: int, 
                          watchlist_performance: dict = None, sector_allocation: dict = None,
                          risk_metrics: dict = None, market_context: dict = None):
        """Send end-of-day summary with enhanced information.

        Parameters
        ----------
        positions : list of dict
            Each dict has: ticker, entry_price, current_price, pnl_pct, pnl_dollar, signal_type, zscore, rsi
        daily_pnl : float
            Total PnL for the day
        active_signals : int
            Number of currently open positions
        watchlist_performance : dict, optional
            Performance metrics for the watchlist (win rate, avg return, etc.)
        sector_allocation : dict, optional
            Breakdown of positions by sector
        risk_metrics : dict, optional
            Risk metrics (max drawdown, volatility, Sharpe estimate, etc.)
        market_context : dict, optional
            Market regime, volatility regime, etc.
        """
        pnl_emoji = "✅" if daily_pnl >= 0 else "❌"

        # Header with date and basic stats
        header = (
            f"📋 <b>ENHANCED DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"📊 Active Positions: {active_signals}\n"
            f"💰 Daily PnL: <b>${daily_pnl:+.2f}</b> {pnl_emoji}\n"
        )

        # Add market context if available
        if market_context:
            regime = market_context.get('regime', 'UNKNOWN')
            vol_regime = market_context.get('volatility_regime', 'UNKNOWN')
            header += f"📈 Market Regime: {regime} | Vol: {vol_regime}\n"
        
        header += f"━━━━━━━━━━━━━━━━━━━━━━\n"

        # Watchlist performance section
        body = ""
        if watchlist_performance:
            win_rate = watchlist_performance.get('win_rate', 0)
            avg_return = watchlist_performance.get('avg_return', 0)
            total_trades = watchlist_performance.get('total_trades', 0)
            body += (
                f"<b>Watchlist Performance:</b>\n"
                f"  • Win Rate: {win_rate:.1f}% ({total_trades} trades)\n"
                f"  • Avg Return/Trade: {avg_return:+.2f}%\n"
            )
        
        # Sector allocation section
        if sector_allocation:
            body += f"\n<b>Sector Allocation:</b>\n"
            for sector, count in sector_allocation.items():
                body += f"  • {sector}: {count} position{'s' if count != 1 else ''}\n"

        # Risk metrics section
        if risk_metrics:
            max_dd = risk_metrics.get('max_drawdown', 0)
            sharpe_est = risk_metrics.get('sharpe_estimate', 0)
            vol_percentile = risk_metrics.get('volatility_percentile', 0)
            body += (
                f"\n<b>Risk Metrics:</b>\n"
                f"  • Est. Sharpe: {sharpe_est:.2f}\n"
                f"  • Vol Percentile: {vol_percentile:.0f}%\n"
                f"  • Max DD Est: {max_dd:.2f}%\n"
            )

        # Open positions section
        if positions:
            body += f"\n<b>Open Positions ({len(positions)}):</b>\n"
            for pos in positions:
                pnl_sign = "+" if pos["pnl_dollar"] >= 0 else ""
                signal_emoji = "🟢" if "LONG" in pos["signal_type"] else "🔴"
                
                # Add enhanced position details
                zscore_info = f" | Z:{pos.get('zscore', 0):+.1f}" if 'zscore' in pos else ""
                rsi_info = f" | RSI:{pos.get('rsi', 50):.0f}" if 'rsi' in pos else ""
                
                body += (
                    f"  {signal_emoji} <b>{pos['ticker']}</b> ({pos['signal_type']})\n"
                    f"    Entry: ${pos['entry_price']:.2f} → Current: ${pos['current_price']:.2f}{zscore_info}{rsi_info}\n"
                    f"    PnL: {pnl_sign}{pos['pnl_pct']:.2f}% (${pnl_sign}{pos['pnl_dollar']:.2f})\n"
                )
        else:
            body += "  <i>No open positions</i>\n"

        footer = f"━━━━━━━━━━━━━━━━━━━━━━\n"

        message = header + body + footer
        return self.send_message(message)

    def send_test_message(self):
        """Send a test message to verify connectivity."""
        message = (
            "✅ <b>Telegram Alert Bot is online!</b>\n"
            f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"📋 Watchlist: {', '.join(WATCHLIST)}\n"
            f"🔄 Check interval: 30 min during market hours"
        )
        return self.send_message(message)


# ---------------------------------------------------------------------------
# CLI entry-point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Alert Bot for Trading Signals")
    parser.add_argument("--token", type=str, default=None, help="Telegram Bot Token")
    parser.add_argument("--chat-id", type=str, default=None, help="Telegram Chat ID")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        bot = TelegramBot(token=args.token, chat_id=args.chat_id)

        if args.test:
            result = bot.send_test_message()
            if result.get("ok"):
                print("✅ Test message sent successfully!")
            else:
                print(f"❌ Failed to send test message: {result}")
                sys.exit(1)
        else:
            print("Use --test to send a test message.")
            print(f"Watchlist: {WATCHLIST}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables, or use --token/--chat-id flags.")
        sys.exit(1)
