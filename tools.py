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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# RESEARCH AGENT TOOLS
# ═══════════════════════════════════════════════════════════════════

def get_stock_data(ticker: str) -> str:
    """
    Fetch key financial data for a stock ticker.
    Returns price, market cap, PE ratio, sector, 52-week range, volume,
    and other fundamental metrics.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            try:
                fast = stock.fast_info
                price = getattr(fast, "last_price", None)
                market_cap = getattr(fast, "market_cap", None)
            except Exception:
                return json.dumps({"error": f"Could not retrieve data for '{ticker}'."})

            if price is None:
                return json.dumps({"error": f"No data found for '{ticker}'."})

            return json.dumps({
                "ticker": ticker.upper(),
                "price": round(price, 2),
                "market_cap": market_cap,
                "note": "Limited data available via fallback."
            }, indent=2)

        data = {
            "ticker": ticker.upper(),
            "name": info.get("shortName", "N/A"),
            "price": info.get("regularMarketPrice") or info.get("currentPrice", "N/A"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "50_day_avg": info.get("fiftyDayAverage", "N/A"),
            "200_day_avg": info.get("twoHundredDayAverage", "N/A"),
            "volume": info.get("volume", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "revenue": info.get("totalRevenue", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "free_cash_flow": info.get("freeCashflow", "N/A"),
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
    """Calculate current returns for every holding in the portfolio."""
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
            cost_basis = holding["cost_basis"]

            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get("regularMarketPrice") or info.get("currentPrice")

                if current_price is None:
                    fast = stock.fast_info
                    current_price = getattr(fast, "last_price", None)

                if current_price is None:
                    results.append({"ticker": ticker, "error": "Could not fetch price"})
                    continue

                cost_total = shares * cost_basis
                current_total = shares * current_price
                gain_loss = current_total - cost_total
                return_pct = ((current_price - cost_basis) / cost_basis) * 100

                total_cost += cost_total
                total_current += current_total

                results.append({
                    "ticker": ticker,
                    "shares": shares,
                    "cost_basis": round(cost_basis, 2),
                    "current_price": round(current_price, 2),
                    "cost_total": round(cost_total, 2),
                    "current_total": round(current_total, 2),
                    "gain_loss": round(gain_loss, 2),
                    "return_pct": round(return_pct, 2),
                })
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        total_gain = total_current - total_cost
        total_return_pct = ((total_current - total_cost) / total_cost * 100) if total_cost > 0 else 0

        return json.dumps({
            "holdings": results,
            "summary": {
                "total_invested": round(total_cost, 2),
                "total_current_value": round(total_current, 2),
                "total_gain_loss": round(total_gain, 2),
                "total_return_pct": round(total_return_pct, 2),
            },
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


def add_to_portfolio(ticker: str, shares: float, cost_basis: float) -> str:
    """Add a stock to the portfolio or update an existing position."""
    portfolio = load_portfolio()
    ticker = ticker.upper().strip()

    for holding in portfolio:
        if holding["ticker"] == ticker:
            old_shares = holding["shares"]
            old_cost = holding["cost_basis"]
            new_total = old_shares + shares
            holding["cost_basis"] = round(
                (old_shares * old_cost + shares * cost_basis) / new_total, 4
            )
            holding["shares"] = new_total
            save_portfolio(portfolio)
            return f"Updated {ticker}: now {new_total} shares at avg cost ${holding['cost_basis']:.2f}"

    portfolio.append({"ticker": ticker, "shares": shares, "cost_basis": round(cost_basis, 4)})
    save_portfolio(portfolio)
    return f"Added {ticker}: {shares} shares at ${cost_basis:.2f}"


def remove_from_portfolio(ticker: str) -> str:
    """Remove a stock from the portfolio."""
    portfolio = load_portfolio()
    ticker = ticker.upper().strip()
    new_portfolio = [h for h in portfolio if h["ticker"] != ticker]
    if len(new_portfolio) == len(portfolio):
        return f"{ticker} not found in portfolio."
    save_portfolio(new_portfolio)
    return f"Removed {ticker} from portfolio."
