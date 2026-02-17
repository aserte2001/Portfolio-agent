"""
Custom tool functions for the AI Portfolio agents.

Each function serves as a tool that agents can invoke via OpenAI function calling.
Tools are pure Python functions that wrap external APIs (yfinance) and local data.
"""

import json
import logging
from datetime import datetime
from typing import Any

import yfinance as yf

from config import PORTFOLIO_PATH
from currency_converter import (
    usd_to_eur, format_eur, get_usd_to_eur_rate,
    to_eur, detect_currency,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# RESEARCH AGENT TOOLS
# ═══════════════════════════════════════════════════════════════════

def _get_price_robust(stock: yf.Ticker, ticker: str) -> float | None:
    """Get current price with multiple fallback strategies."""
    # Strategy 1: fast_info
    try:
        price = getattr(stock.fast_info, "last_price", None)
        if price is not None and price > 0:
            return round(price, 2)
    except Exception:
        pass

    # Strategy 2: info dict
    try:
        info = stock.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is not None and price > 0:
            return round(price, 2)
    except Exception:
        pass

    # Strategy 3: history (most reliable)
    try:
        hist = stock.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].dropna().iloc[-1])
            if price > 0:
                return round(price, 2)
    except Exception:
        pass

    logger.warning(f"_get_price_robust: all strategies failed for {ticker}")
    return None


def get_stock_data(ticker: str) -> str:
    """
    Fetch key financial data for a stock ticker.
    Returns price, market cap, PE ratio, sector, 52-week range, volume,
    and other fundamental metrics. Uses multiple fallback strategies.
    Automatically detects the ticker's native currency and only converts
    to EUR when necessary (e.g. USD tickers). EUR-denominated tickers
    (like IROB.DE) are returned without conversion.
    """
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)

        # Get price first with robust fallback
        price = _get_price_robust(stock, ticker)

        # Try to get full info
        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        # If info has no price, inject our robust price
        if price is None:
            price = info.get("regularMarketPrice") or info.get("currentPrice")

        if price is None:
            return json.dumps({"error": f"No price data found for '{ticker}'."})

        # Detect native currency
        native_currency = detect_currency(info)
        is_native_eur = (native_currency == "EUR")

        # Get market cap with fallback
        market_cap = info.get("marketCap")
        if not market_cap:
            try:
                market_cap = getattr(stock.fast_info, "market_cap", None)
            except Exception:
                market_cap = None

        def _conv(val):
            """Convert a value to EUR, respecting the native currency."""
            if isinstance(val, (int, float)):
                return round(to_eur(val, native_currency), 2)
            return val

        rate = get_usd_to_eur_rate()

        price_eur = round(to_eur(price, native_currency), 2)

        conversion_note = (
            "Ticker notiert in EUR – keine Umrechnung noetig"
            if is_native_eur
            else f"Alle Preise von {native_currency} in EUR umgerechnet (Kurs: 1 USD = {rate:.4f} EUR)"
        )

        data = {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "price_eur": price_eur,
            "price_native": price,
            "native_currency": native_currency,
            "currency": "EUR",
            "exchange_rate": 1.0 if is_native_eur else round(rate, 4),
            "needs_conversion": not is_native_eur,
            "market_cap_eur": _conv(market_cap) if market_cap else "N/A",
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "eps_eur": _conv(info.get("trailingEps")) if info.get("trailingEps") else "N/A",
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "52_week_high_eur": _conv(info.get("fiftyTwoWeekHigh")) if info.get("fiftyTwoWeekHigh") else "N/A",
            "52_week_low_eur": _conv(info.get("fiftyTwoWeekLow")) if info.get("fiftyTwoWeekLow") else "N/A",
            "50_day_avg_eur": _conv(info.get("fiftyDayAverage")) if info.get("fiftyDayAverage") else "N/A",
            "200_day_avg_eur": _conv(info.get("twoHundredDayAverage")) if info.get("twoHundredDayAverage") else "N/A",
            "volume": info.get("volume", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "revenue_eur": _conv(info.get("totalRevenue")) if info.get("totalRevenue") else "N/A",
            "profit_margin": info.get("profitMargins", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "free_cash_flow_eur": _conv(info.get("freeCashflow")) if info.get("freeCashflow") else "N/A",
            "note": conversion_note,
        }
        return json.dumps(data, indent=2, default=str)

    except Exception as e:
        logger.error(f"get_stock_data failed for {ticker}: {e}")
        return json.dumps({"error": f"Error fetching stock data: {str(e)}"})


def get_company_info(ticker: str) -> str:
    """
    Fetch detailed company information including business summary,
    employees, website, and key officers.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info

        data = {
            "ticker": ticker.upper(),
            "name": info.get("shortName", "N/A"),
            "summary": info.get("longBusinessSummary", "No summary available."),
            "website": info.get("website", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"),
            "headquarters": f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}",
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }

        officers = info.get("companyOfficers", [])
        if officers:
            data["key_officers"] = [
                {"name": o.get("name", "N/A"), "title": o.get("title", "N/A")}
                for o in officers[:5]
            ]

        return json.dumps(data, indent=2, default=str)

    except Exception as e:
        logger.error(f"get_company_info failed for {ticker}: {e}")
        return json.dumps({"error": f"Error fetching company info: {str(e)}"})


# ═══════════════════════════════════════════════════════════════════
# NEWS AGENT TOOLS
# ═══════════════════════════════════════════════════════════════════

def search_news(ticker: str) -> str:
    """
    Retrieve recent news headlines for a stock ticker using yfinance.
    Returns titles, publishers, links, and timestamps.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        news_items = stock.news

        if not news_items:
            return json.dumps({
                "ticker": ticker.upper(),
                "news": [],
                "message": "No recent news found."
            }, indent=2)

        articles = []
        for item in news_items[:10]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            article = {
                "title": content.get("title") or item.get("title", "N/A"),
                "publisher": (content.get("provider", {}).get("displayName")
                              or item.get("publisher", "N/A")),
                "link": (content.get("canonicalUrl", {}).get("url")
                         or item.get("link", "N/A")),
                "published": (content.get("pubDate")
                              or item.get("providerPublishTime", "N/A")),
                "summary": content.get("summary", ""),
            }
            articles.append(article)

        return json.dumps({
            "ticker": ticker.upper(),
            "news_count": len(articles),
            "articles": articles,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_news failed for {ticker}: {e}")
        return json.dumps({"error": f"Error fetching news: {str(e)}"})


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO MONITOR TOOLS
# ═══════════════════════════════════════════════════════════════════

def get_portfolio_data() -> str:
    """Read all holdings from portfolio.json."""
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            portfolio = json.load(f)
        if not portfolio:
            return json.dumps({"holdings": [], "message": "Portfolio is empty."})
        return json.dumps({"holdings": portfolio, "count": len(portfolio)}, indent=2)
    except FileNotFoundError:
        return json.dumps({"holdings": [], "message": "No portfolio file found."})
    except Exception as e:
        logger.error(f"get_portfolio_data failed: {e}")
        return json.dumps({"error": f"Error reading portfolio: {str(e)}"})


def calculate_returns() -> str:
    """Calculate current returns for every holding in the portfolio.
    Automatically detects each ticker's native currency and only converts
    to EUR when necessary."""
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            portfolio = json.load(f)

        if not portfolio:
            return json.dumps({"message": "Portfolio is empty."})

        results = []
        total_cost = 0.0
        total_current = 0.0

        for holding in portfolio:
            ticker = holding["ticker"]
            shares = holding["shares"]
            cost_basis_eur = holding.get("cost_basis_eur", holding.get("cost_basis", 0))

            try:
                stock = yf.Ticker(ticker)
                price_raw = _get_price_robust(stock, ticker)

                if price_raw is None:
                    results.append({"ticker": ticker, "error": "Could not fetch price"})
                    continue

                # Detect native currency to avoid double conversion
                info = {}
                try:
                    info = stock.info or {}
                except Exception:
                    pass
                native_currency = detect_currency(info)

                current_price_eur = to_eur(price_raw, native_currency)
                cost_total = shares * cost_basis_eur
                current_total = shares * current_price_eur
                gain_loss = current_total - cost_total
                return_pct = ((current_price_eur - cost_basis_eur) / cost_basis_eur) * 100 if cost_basis_eur > 0 else 0

                total_cost += cost_total
                total_current += current_total

                results.append({
                    "ticker": ticker,
                    "shares": shares,
                    "cost_basis_eur": round(cost_basis_eur, 2),
                    "current_price_eur": round(current_price_eur, 2),
                    "native_currency": native_currency,
                    "cost_total_eur": round(cost_total, 2),
                    "current_total_eur": round(current_total, 2),
                    "gain_loss_eur": round(gain_loss, 2),
                    "return_pct": round(return_pct, 2),
                })
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        total_gain = total_current - total_cost
        total_return_pct = ((total_current - total_cost) / total_cost * 100) if total_cost > 0 else 0

        return json.dumps({
            "holdings": results,
            "currency": "EUR",
            "exchange_rate": round(get_usd_to_eur_rate(), 4),
            "summary": {
                "total_invested_eur": round(total_cost, 2),
                "total_current_value_eur": round(total_current, 2),
                "total_gain_loss_eur": round(total_gain, 2),
                "total_return_pct": round(total_return_pct, 2),
            },
            "note": "Alle Werte in EUR (automatische Waehrungserkennung aktiv)",
            "timestamp": datetime.now().isoformat(),
        }, indent=2)

    except FileNotFoundError:
        return json.dumps({"message": "No portfolio file found."})
    except Exception as e:
        logger.error(f"calculate_returns failed: {e}")
        return json.dumps({"error": f"Error calculating returns: {str(e)}"})


# ═══════════════════════════════════════════════════════════════════
# TOOL REGISTRY – maps function names to callables and OpenAI schemas
# ═══════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS: dict[str, Any] = {
    "get_stock_data": get_stock_data,
    "get_company_info": get_company_info,
    "search_news": search_news,
    "get_portfolio_data": get_portfolio_data,
    "calculate_returns": calculate_returns,
}

TOOL_SCHEMAS = {
    "research": [
        {
            "type": "function",
            "function": {
                "name": "get_stock_data",
                "description": "Fetch key financial metrics for a stock: price, market cap, PE ratio, revenue, margins, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, RKLB)"}
                    },
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_company_info",
                "description": "Fetch detailed company information: business summary, employees, officers, sector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"],
                },
            },
        },
    ],
    "news": [
        {
            "type": "function",
            "function": {
                "name": "search_news",
                "description": "Retrieve recent news articles for a stock ticker with headlines, publishers, and summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"],
                },
            },
        },
    ],
    "portfolio": [
        {
            "type": "function",
            "function": {
                "name": "get_portfolio_data",
                "description": "Read all current holdings from the portfolio (tickers, shares, cost basis).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_returns",
                "description": "Calculate live returns for all portfolio holdings: current value, gain/loss, return percentage.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO MANAGEMENT (used by the Streamlit UI, not by agents)
# ═══════════════════════════════════════════════════════════════════

def load_portfolio() -> list[dict[str, Any]]:
    """Load portfolio from JSON file."""
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_portfolio(portfolio: list[dict[str, Any]]) -> None:
    """Save portfolio to JSON file."""
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)


def add_to_portfolio(ticker: str, shares: float, cost_basis_eur: float) -> str:
    """Add a stock to the portfolio or update an existing position. Cost basis in EUR."""
    portfolio = load_portfolio()
    ticker = ticker.upper().strip()

    for holding in portfolio:
        if holding["ticker"] == ticker:
            old_shares = holding["shares"]
            old_cost = holding.get("cost_basis_eur", holding.get("cost_basis", 0))
            new_total = old_shares + shares
            holding["cost_basis_eur"] = round(
                (old_shares * old_cost + shares * cost_basis_eur) / new_total, 4
            )
            holding["shares"] = new_total
            holding.pop("cost_basis", None)
            save_portfolio(portfolio)
            return f"{ticker} aktualisiert: {new_total} Anteile zu \u20ac{holding['cost_basis_eur']:.2f} Durchschnitt"

    portfolio.append({
        "ticker": ticker,
        "shares": shares,
        "cost_basis_eur": round(cost_basis_eur, 4),
        "date_added": datetime.now().strftime("%Y-%m-%d"),
    })
    save_portfolio(portfolio)
    return f"{ticker} hinzugefuegt: {shares} Anteile zu \u20ac{cost_basis_eur:.2f}"


def remove_from_portfolio(ticker: str) -> str:
    """Remove a stock from the portfolio."""
    portfolio = load_portfolio()
    ticker = ticker.upper().strip()
    new_portfolio = [h for h in portfolio if h["ticker"] != ticker]
    if len(new_portfolio) == len(portfolio):
        return f"{ticker} nicht im Portfolio gefunden."
    save_portfolio(new_portfolio)
    return f"{ticker} aus dem Portfolio entfernt."
