"""
USD → EUR Currency Converter with caching and fallback.

Uses the free exchangerate-api.com endpoint (no API key required).
The exchange rate is cached for 1 hour to avoid excessive API calls.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache (works across Streamlit reruns within the same process)
_cache: dict = {
    "rate": None,
    "timestamp": 0.0,
}

CACHE_TTL_SECONDS = 3600  # 1 hour
FALLBACK_RATE = 0.92
API_URL = "https://api.exchangerate-api.com/v4/latest/USD"
API_TIMEOUT = 5


def get_usd_to_eur_rate() -> float:
    """
    Fetch current USD→EUR exchange rate.
    Returns cached rate if less than 1 hour old.
    Falls back to 0.92 if the API is unreachable.
    """
    now = time.time()

    # Return cached rate if still fresh
    if _cache["rate"] is not None and (now - _cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _cache["rate"]

    # Try fetching live rate
    try:
        import requests
        resp = requests.get(API_URL, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"]["EUR"]

        if isinstance(rate, (int, float)) and rate > 0:
            _cache["rate"] = rate
            _cache["timestamp"] = now
            logger.info(f"[FX] Live rate fetched: 1 USD = {rate:.4f} EUR")
            return rate

    except Exception as e:
        logger.warning(f"[FX] API call failed: {e}")

    # Return previous cached rate if available, otherwise fallback
    if _cache["rate"] is not None:
        logger.info(f"[FX] Using stale cached rate: {_cache['rate']:.4f}")
        return _cache["rate"]

    logger.warning(f"[FX] Using fallback rate: {FALLBACK_RATE}")
    return FALLBACK_RATE


def usd_to_eur(usd_amount: Optional[float]) -> Optional[float]:
    """Convert a USD amount to EUR. Returns None if input is None."""
    if usd_amount is None:
        return None
    rate = get_usd_to_eur_rate()
    return usd_amount * rate


def format_eur(amount: Optional[float]) -> str:
    """
    Format a number as Euro with European style: €1.234,56
    Returns 'N/A' if amount is None.
    """
    if amount is None:
        return "N/A"

    negative = amount < 0
    abs_val = abs(amount)
    suffix = ""

    if abs_val >= 1_000_000_000:
        num_str = f"{abs_val / 1_000_000_000:,.2f}"
        suffix = " Mrd."
    elif abs_val >= 1_000_000:
        num_str = f"{abs_val / 1_000_000:,.2f}"
        suffix = " Mio."
    else:
        num_str = f"{abs_val:,.2f}"

    # Convert to European format: 1,234.56 → 1.234,56
    num_str = num_str.replace(",", "X").replace(".", ",").replace("X", ".")

    prefix = "-" if negative else ""
    return f"{prefix}\u20ac{num_str}{suffix}"


def to_eur(amount: Optional[float], source_currency: str = "USD") -> Optional[float]:
    """
    Convert an amount to EUR, respecting the source currency.
    If the source is already EUR, the amount is returned unchanged.
    Supports USD and other major currencies via the exchange-rate API.
    """
    if amount is None:
        return None
    src = source_currency.upper().strip()
    if src == "EUR":
        return amount
    if src == "USD":
        return usd_to_eur(amount)
    # For other currencies, try fetching the specific rate
    try:
        import requests
        resp = requests.get(
            f"https://api.exchangerate-api.com/v4/latest/{src}",
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        rate = resp.json()["rates"]["EUR"]
        return amount * rate
    except Exception:
        logger.warning(f"[FX] Could not convert {src}→EUR, falling back to USD path")
        return usd_to_eur(amount)


def detect_currency(ticker_info: dict) -> str:
    """
    Detect the trading currency from a yfinance info dict.
    Returns the ISO currency code (e.g. 'USD', 'EUR', 'GBP').
    """
    currency = ticker_info.get("currency", "")
    if currency:
        return currency.upper().strip()
    # Heuristic: tickers ending in .DE / .F / .PA etc. are often EUR
    return "USD"


def is_using_fallback() -> bool:
    """Check if we're using a fallback/estimated rate."""
    return _cache["rate"] is None
