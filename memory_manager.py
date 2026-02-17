"""
Investment Memory System – persistent user profile management.

Loads, saves, and summarizes the user's investment profile so that
all agents can personalize their analysis and recommendations.
The profile is injected into every agent's system prompt, making
the agents "remember" and adapt to the user's preferences.
"""

import json
import logging
from datetime import datetime
from typing import Any

from config import PROFILE_PATH

logger = logging.getLogger(__name__)

# ── Default profile structure ──────────────────────────────────────
DEFAULT_PROFILE: dict[str, Any] = {
    "risk_tolerance": "Mittel",
    "investment_horizon": "5-10 Jahre",
    "preferred_sectors": [],
    "stock_size_preference": "Alle Groessen",
    "geographic_focus": "USA",
    "max_position_size_pct": 20,
    "dividend_preference": False,
    "philosophy": "",
    "admired_stocks": "",
    "last_updated": None,
}

# ── Sector options (German labels) ────────────────────────────────
SECTOR_OPTIONS = [
    "\U0001f680 Raumfahrttechnologie",
    "\U0001f916 Robotik & Automatisierung",
    "\U0001f9e0 Kuenstliche Intelligenz",
    "\u2600\ufe0f Erneuerbare Energien",
    "\U0001f4bb Software / SaaS",
    "\U0001f3e5 Gesundheit / Biotech",
    "\U0001f4b0 Fintech",
    "\U0001f3ae Gaming / Unterhaltung",
    "\U0001f6e1\ufe0f Cybersicherheit",
    "\u26d3\ufe0f Blockchain / Krypto",
    "\U0001f4e6 E-Commerce",
    "\U0001f3ed Industrie / Fertigung",
]

RISK_LEVELS = ["Niedrig", "Mittel", "Hoch", "Sehr hoch"]
HORIZONS = ["1-3 Jahre", "3-5 Jahre", "5-10 Jahre", "10+ Jahre"]
STOCK_SIZES = ["Large Cap (>$10 Mrd.)", "Mid Cap ($2-$10 Mrd.)", "Small Cap ($300M-$2 Mrd.)", "Micro Cap (<$300M)", "Alle Groessen"]
GEO_OPTIONS = ["USA", "Europa", "Asien", "Global"]


def load_profile() -> dict[str, Any]:
    """Load the investment profile from disk, or return defaults."""
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
        merged = {**DEFAULT_PROFILE, **profile}
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_PROFILE.copy()


def save_profile(profile: dict[str, Any]) -> None:
    """Save the investment profile to disk with a timestamp."""
    profile["last_updated"] = datetime.now().isoformat()
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info("Investment profile saved.")


def reset_profile() -> None:
    """Reset the profile to defaults."""
    save_profile(DEFAULT_PROFILE.copy())
    logger.info("Investment profile reset to defaults.")


def is_profile_configured() -> bool:
    """Check if the user has customized their profile beyond defaults."""
    profile = load_profile()
    return bool(
        profile.get("preferred_sectors")
        or profile.get("philosophy")
        or profile.get("risk_tolerance") not in ("Mittel", "Medium")
        or profile.get("investment_horizon") not in ("5-10 Jahre", "5-10y")
    )


def get_profile_summary() -> str:
    """
    Generate a concise one-line summary of the user profile.
    Used in the sidebar badge.
    """
    profile = load_profile()
    if not is_profile_configured():
        return "Nicht konfiguriert"

    parts = []
    risk = profile.get("risk_tolerance", "Mittel")
    risk_emoji = {
        "Niedrig": "\U0001f6e1\ufe0f",
        "Mittel": "\u2696\ufe0f",
        "Hoch": "\u26a1",
        "Sehr hoch": "\U0001f525",
    }
    parts.append(f"{risk_emoji.get(risk, '')} {risk}")

    horizon = profile.get("investment_horizon", "")
    if horizon:
        parts.append(horizon)

    sectors = profile.get("preferred_sectors", [])
    if sectors:
        short = [s.split(" ")[0] for s in sectors[:2]]
        suffix = f" +{len(sectors) - 2}" if len(sectors) > 2 else ""
        parts.append("".join(short) + suffix)

    return " · ".join(parts)


def get_profile_context() -> str:
    """
    Generate a detailed context string to inject into agent system prompts.
    This is the core of the memory system: agents use this to personalize
    every analysis and recommendation. Written in German so agents respond in German.
    """
    profile = load_profile()

    if not is_profile_configured():
        return ""

    lines = ["ANLAGEPROFIL DES NUTZERS (nutze dies zur Personalisierung deiner Analyse):"]

    risk = profile.get("risk_tolerance", "Mittel")
    lines.append(f"- Risikobereitschaft: {risk}")

    horizon = profile.get("investment_horizon", "5-10 Jahre")
    lines.append(f"- Anlagehorizont: {horizon}")

    sectors = profile.get("preferred_sectors", [])
    if sectors:
        lines.append(f"- Bevorzugte Sektoren: {', '.join(sectors)}")

    size = profile.get("stock_size_preference", "Alle Groessen")
    lines.append(f"- Unternehmensgroesse: {size}")

    geo = profile.get("geographic_focus", "USA")
    lines.append(f"- Geografischer Fokus: {geo}")

    max_pos = profile.get("max_position_size_pct", 20)
    lines.append(f"- Max. Positionsgroesse: {max_pos}% des Portfolios")

    div_pref = profile.get("dividend_preference", False)
    lines.append(f"- Dividendenpraeferenz: {'Ja' if div_pref else 'Nein'}")

    philosophy = profile.get("philosophy", "")
    if philosophy:
        lines.append(f"- Anlagephilosophie: \"{philosophy}\"")

    admired = profile.get("admired_stocks", "")
    if admired:
        lines.append(f"- Bewunderte Investments: \"{admired}\"")

    lines.append("\nBasierend auf diesem Profil:")

    if risk in ("Hoch", "Sehr hoch", "High", "Very High"):
        lines.append("- Betone langfristiges Wachstumspotenzial statt kurzfristiger Volatilitaet.")
        lines.append("- Keine Panik bei 20-30% Kursrueckgaengen; fokussiere auf Fundamentaldaten.")
    elif risk in ("Niedrig", "Low"):
        lines.append("- Priorisiere Kapitalerhalt und stabile Renditen.")
        lines.append("- Hebe Hochvolatilitaetsrisiken deutlich hervor.")

    if "10+" in horizon:
        lines.append("- Fokussiere auf langjaehrige Trends und saekulare Wachstumsthemen.")
        lines.append("- Kurzfristige Gewinnverfehlungen sind weniger wichtig als die langfristige Entwicklung.")
    elif "1-3" in horizon:
        lines.append("- Fokussiere auf kurzfristige Katalysatoren und aktuelle Bewertung.")

    if any("Raumfahrt" in s or "Space" in s for s in sectors):
        lines.append("- Der Nutzer interessiert sich fuer Raumfahrttechnologie; hebe Luft- und Raumfahrtunternehmen hervor.")
    if any("KI" in s or "Intelligenz" in s or "AI" in s for s in sectors):
        lines.append("- Der Nutzer interessiert sich fuer KI; hebe KI-Infrastruktur, Modelle und Anwendungen hervor.")

    return "\n".join(lines)
