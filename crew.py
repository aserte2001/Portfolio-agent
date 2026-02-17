"""
Crew orchestration module – multi-agent workflow coordination.

This module wires agents and tasks together into executable workflows,
mirroring the "Crew" concept from frameworks like CrewAI. Each workflow
function creates the right agent, assigns a task, and returns the result.

Demonstrates: agent instantiation, task routing, and sequential orchestration.
"""

import logging

from agents import (
    create_research_agent,
    create_news_agent,
    create_portfolio_monitor_agent,
)
from tasks import (
    stock_analysis_task,
    news_analysis_task,
    portfolio_analysis_task,
)

logger = logging.getLogger(__name__)


def run_stock_analysis(ticker: str, model: str | None = None) -> str:
    """
    Single-agent workflow: Research Agent analyzes a stock.

    Flow: Create Agent → Assign Task → Agent calls tools → Returns report
    """
    logger.info(f"[Crew] Stock analysis workflow started for {ticker}")
    agent = create_research_agent(model=model)
    task = stock_analysis_task(ticker)
    result = agent.execute(task, model_override=model)
    logger.info(f"[Crew] Stock analysis workflow complete for {ticker}")
    return result


def run_news_analysis(ticker: str, model: str | None = None) -> str:
    """
    Single-agent workflow: News Agent fetches and summarizes news.

    Flow: Create Agent → Assign Task → Agent calls tools → Returns briefing
    """
    logger.info(f"[Crew] News analysis workflow started for {ticker}")
    agent = create_news_agent(model=model)
    task = news_analysis_task(ticker)
    result = agent.execute(task, model_override=model)
    logger.info(f"[Crew] News analysis workflow complete for {ticker}")
    return result


def run_portfolio_analysis(model: str | None = None) -> str:
    """
    Single-agent workflow: Portfolio Monitor analyzes all holdings.

    Flow: Create Agent → Assign Task → Agent calls tools → Returns health report
    """
    logger.info("[Crew] Portfolio analysis workflow started")
    agent = create_portfolio_monitor_agent(model=model)
    task = portfolio_analysis_task()
    result = agent.execute(task, model_override=model)
    logger.info("[Crew] Portfolio analysis workflow complete")
    return result


def run_full_analysis(ticker: str, model: str | None = None) -> dict[str, str]:
    """
    Multi-agent sequential workflow: all 3 agents collaborate.

    Flow: Research Agent → News Agent → Portfolio Monitor → Combined report
    Demonstrates agent coordination where each agent contributes its specialty.
    """
    logger.info(f"[Crew] Full multi-agent workflow started for {ticker}")

    research_result = run_stock_analysis(ticker, model=model)
    news_result = run_news_analysis(ticker, model=model)
    portfolio_result = run_portfolio_analysis(model=model)

    logger.info(f"[Crew] Full multi-agent workflow complete for {ticker}")

    return {
        "research": research_result,
        "news": news_result,
        "portfolio": portfolio_result,
    }
