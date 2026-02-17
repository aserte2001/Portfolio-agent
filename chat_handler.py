"""
Chat Handler – intelligent question routing and conversation management.

Routes user questions to the appropriate agent(s) based on intent classification,
manages conversation history, and provides a unified chat interface.
Demonstrates: intent classification, agent routing, context management.
"""

import json
import logging
from datetime import datetime
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_HISTORY_PATH, MAX_CHAT_MESSAGES
from agents import create_research_agent, create_news_agent, create_portfolio_monitor_agent
from memory_manager import get_profile_context

logger = logging.getLogger(__name__)

# Agent display metadata (German labels)
AGENT_META = {
    "research": {"name": "Research Agent", "emoji": "\U0001f50d", "color": "#667eea"},
    "news":     {"name": "News Agent",     "emoji": "\U0001f4f0", "color": "#f59e0b"},
    "portfolio":{"name": "Portfolio Monitor", "emoji": "\U0001f4ca", "color": "#10b981"},
    "router":   {"name": "KI-Router",      "emoji": "\U0001f916", "color": "#888"},
}


# ═══════════════════════════════════════════════════════════════════
# CHAT HISTORY PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

def load_chat_history() -> list[dict[str, Any]]:
    """Load chat history from disk."""
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history[-MAX_CHAT_MESSAGES:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_chat_history(history: list[dict[str, Any]]) -> None:
    """Save chat history to disk (limited to MAX_CHAT_MESSAGES)."""
    trimmed = history[-MAX_CHAT_MESSAGES:]
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, default=str, ensure_ascii=False)


def clear_chat_history() -> None:
    """Clear all chat history."""
    save_chat_history([])


def add_message(history: list[dict], role: str, content: str,
                agent_type: str | None = None) -> list[dict]:
    """Append a message to the history and return updated history."""
    msg: dict[str, Any] = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    if agent_type:
        msg["agent_type"] = agent_type
    history.append(msg)
    return history


# ═══════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION – routes questions to the right agent
# ═══════════════════════════════════════════════════════════════════

def classify_intent(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Use GPT to classify the user's question intent.
    Returns one of: 'research', 'news', 'portfolio', 'multi'.
    Falls back to keyword matching if the API call fails.
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Fragenklassifikator fuer eine Investment-App. "
                        "Der Nutzer stellt Fragen auf Deutsch oder Englisch. "
                        "Klassifiziere die Frage in genau EINE Kategorie:\n"
                        "- 'research': Fragen zu bestimmten Aktien, Fundamentaldaten, "
                        "  Kaufen/Verkaufen/Halten, Analyse, Bewertung, Kursziele\n"
                        "- 'news': Fragen zu aktuellen Nachrichten, Ereignissen, "
                        "  was heute/kuerzlich passiert ist\n"
                        "- 'portfolio': Fragen zum Portfolio des Nutzers, Positionen, "
                        "  Rendite, Risiko, Diversifikation, Rebalancing\n"
                        "- 'multi': Komplexe Fragen, die mehrere Agenten benoetigen\n\n"
                        "Antworte NUR mit dem Kategoriewort, sonst nichts."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=10,
        )
        intent = response.choices[0].message.content.strip().lower()
        if intent in ("research", "news", "portfolio", "multi"):
            logger.info(f"[Router] Classified intent: {intent}")
            return intent
    except Exception as e:
        logger.warning(f"[Router] GPT classification failed: {e}, falling back to keywords")

    return _keyword_classify(question)


def _keyword_classify(question: str) -> str:
    """Fallback keyword-based classification (German + English)."""
    q = question.lower()

    news_words = ["news", "nachrichten", "aktuell", "heute", "kuerzlich",
                  "headline", "schlagzeile", "ereignis", "quartals", "meldung"]
    portfolio_words = ["portfolio", "positionen", "rendite", "performance",
                       "konzentration", "diversifik", "rebalancing", "allokation",
                       "meine aktien", "mein depot", "holdings", "risiko meines"]
    research_words = ["kaufen", "verkaufen", "halten", "kursziel", "analyse",
                      "bewertung", "fundamental", "soll ich", "lohnt sich",
                      "unterbewertet", "ueberbewertet", "buy", "sell", "hold"]

    news_score = sum(1 for w in news_words if w in q)
    portfolio_score = sum(1 for w in portfolio_words if w in q)
    research_score = sum(1 for w in research_words if w in q)

    max_score = max(news_score, portfolio_score, research_score)
    if max_score == 0:
        return "research"

    if news_score == max_score:
        return "news"
    if portfolio_score == max_score:
        return "portfolio"
    return "research"


# ═══════════════════════════════════════════════════════════════════
# CHAT EXECUTION – runs the appropriate agent with conversation context
# ═══════════════════════════════════════════════════════════════════

def _build_chat_task(question: str, agent_type: str, chat_history: list[dict]) -> str:
    """
    Build a conversational task string for the agent, including
    recent chat context and the user's investment profile.
    """
    profile_ctx = get_profile_context()

    recent = [m for m in chat_history[-6:] if m["role"] in ("user", "assistant")]
    context_lines = []
    for msg in recent:
        prefix = "Nutzer" if msg["role"] == "user" else "Agent"
        context_lines.append(f"{prefix}: {msg['content'][:300]}")

    task_parts = []

    if profile_ctx:
        task_parts.append(profile_ctx)

    if context_lines:
        task_parts.append("LETZTER GESPRAECHSVERLAUF:")
        task_parts.extend(context_lines)
        task_parts.append("")

    task_parts.append("AKTUELLE FRAGE DES NUTZERS:")
    task_parts.append(question)
    task_parts.append("")
    task_parts.append(
        "Anweisungen: Antworte gespraechisg aber professionell auf Deutsch. "
        "Nutze deine Tools, um echte Daten abzurufen, wenn relevant. "
        "Sei konkret mit Zahlen und Datenpunkten. "
        "Halte die Antwort fokussiert und praegnant (unter 500 Woerter). "
        "Wenn das Anlageprofil des Nutzers vorhanden ist, passe deine Antwort an seine Praeferenzen an."
    )

    return "\n".join(task_parts)


def handle_chat_message(
    question: str,
    chat_history: list[dict],
    model: str | None = None,
) -> tuple[str, str]:
    """
    Process a user's chat message:
    1. Classify intent -> route to agent
    2. Build task with context and profile
    3. Execute agent
    4. Return (response_text, agent_type)
    """
    intent = classify_intent(question, model=model or "gpt-4o-mini")

    if intent == "multi":
        intent = "research"

    agent_creators = {
        "research": create_research_agent,
        "news": create_news_agent,
        "portfolio": create_portfolio_monitor_agent,
    }
    creator = agent_creators.get(intent, create_research_agent)
    agent = creator(model=model)

    task = _build_chat_task(question, intent, chat_history)

    logger.info(f"[Chat] Routing to {intent} agent")
    try:
        response = agent.execute(task, model_override=model)
    except Exception as e:
        logger.error(f"[Chat] Agent execution failed: {e}")
        response = f"Entschuldigung, bei der Verarbeitung deiner Frage ist ein Fehler aufgetreten: {str(e)}"

    return response, intent
