"""
Lightweight Multi-Agent Framework using OpenAI Function Calling.

This module implements the core agent abstraction: each Agent has a role,
goal, backstory, and a set of tools it can invoke autonomously.
The LLM decides which tools to call and synthesizes the results.

The Memory System injects the user's investment profile into every
agent's system prompt, enabling personalized analysis and recommendations.

Architecture demonstrates the same patterns as CrewAI / LangGraph
but with zero framework overhead – pure OpenAI SDK.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, DEFAULT_MODEL, TEMPERATURE, MAX_ITERATIONS
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from memory_manager import get_profile_context

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """
    A specialized AI agent with a distinct role, goal, and toolset.

    Each agent wraps an OpenAI chat completion loop with function calling.
    The agent autonomously decides which tools to use, interprets results,
    and produces a structured final output.

    Memory Integration: The user's investment profile is automatically
    loaded and injected into the system prompt so every agent "knows"
    the user's preferences, risk tolerance, and investment style.
    """

    role: str
    goal: str
    backstory: str
    tool_group: str  # key into TOOL_SCHEMAS ("research", "news", "portfolio")
    model: str = field(default_factory=lambda: DEFAULT_MODEL)
    max_iterations: int = MAX_ITERATIONS

    @property
    def tools(self) -> list[dict]:
        """OpenAI function-calling tool schemas for this agent."""
        return TOOL_SCHEMAS.get(self.tool_group, [])

    @property
    def system_prompt(self) -> str:
        """
        Construct the system prompt from the agent's identity.
        Automatically injects the user's investment profile (memory system).
        """
        base = (
            f"Du bist ein {self.role}.\n\n"
            f"**Ziel:** {self.goal}\n\n"
            f"**Hintergrund:** {self.backstory}\n\n"
            "Anweisungen:\n"
            "- Nutze die verfuegbaren Tools, um Daten zu sammeln, bevor du Schlussfolgerungen ziehst.\n"
            "- Sei gruendlich, datengetrieben und professionell.\n"
            "- Formatiere deine Antwort in sauberem Markdown.\n"
            "- Nenne immer konkrete Zahlen und Kennzahlen aus den Tools.\n"
            "- Du antwortest IMMER auf Deutsch.\n"
        )

        # Memory injection: personalize with user profile
        profile_ctx = get_profile_context()
        if profile_ctx:
            base += f"\n\n{profile_ctx}\n"

        return base

    def execute(self, task_description: str, model_override: str | None = None) -> str:
        """
        Run the agent on a task. The agent will:
        1. Receive the task as a user message
        2. Decide which tools to call (OpenAI function calling)
        3. Execute tools and feed results back
        4. Repeat until done or max_iterations reached
        5. Return the final text response

        This is the agentic loop – the core pattern behind
        frameworks like CrewAI, LangChain, and AutoGen.
        """
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = model_override or self.model

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_description},
        ]

        logger.info(f"[{self.role}] Starting task with model={model}")

        for iteration in range(self.max_iterations):
            logger.info(f"[{self.role}] Iteration {iteration + 1}/{self.max_iterations}")

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": TEMPERATURE,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # If the model wants to call tools, execute them
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    logger.info(f"[{self.role}] Calling tool: {fn_name}({fn_args})")

                    fn = TOOL_FUNCTIONS.get(fn_name)
                    if fn:
                        try:
                            result = fn(**fn_args)
                        except Exception as e:
                            result = json.dumps({"error": str(e)})
                            logger.error(f"[{self.role}] Tool {fn_name} failed: {e}")
                    else:
                        result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                continue

            # No tool calls – the agent is done, return the final message
            final = message.content or ""
            logger.info(f"[{self.role}] Task complete ({iteration + 1} iterations)")
            return final

        logger.warning(f"[{self.role}] Reached max iterations ({self.max_iterations})")
        return messages[-1].get("content", "Agent hat die maximale Anzahl an Iterationen erreicht.")


# ═══════════════════════════════════════════════════════════════════
# PRE-CONFIGURED AGENTS
# ═══════════════════════════════════════════════════════════════════

def create_research_agent(model: str | None = None) -> Agent:
    """Senior Investment Analyst – deep fundamental analysis."""
    return Agent(
        role="Senior Investment Analyst",
        goal=(
            "Liefere tiefgehende, umsetzbare Fundamentalanalysen von Aktien. "
            "Bewerte die finanzielle Gesundheit, das Wachstumspotenzial, die Wettbewerbsposition "
            "und Risikofaktoren, um eine klare Anlageempfehlung abzugeben."
        ),
        backstory=(
            "Du bist ein erfahrener Investment-Analyst mit ueber 15 Jahren Erfahrung an der Wall Street. "
            "Du bist spezialisiert auf die Identifikation von High-Growth-Chancen, insbesondere in den "
            "Bereichen Raumfahrttechnologie, Robotik, KI und aufstrebende Tech-Sektoren. Du hast ein "
            "scharfes Auge fuer Micro-Cap-Wachstumsaktien unter $10 mit 10x-Potenzial und "
            "konzentrierst dich auf 5-10 Jahre Anlagehorizont. Deine Analysen sind gruendlich, "
            "datenbasiert und bei Portfoliomanagern hoch angesehen. "
            "Du antwortest immer auf Deutsch."
        ),
        tool_group="research",
        model=model or DEFAULT_MODEL,
    )


def create_news_agent(model: str | None = None) -> Agent:
    """Financial News Analyst – news monitoring and impact assessment."""
    return Agent(
        role="Finanz-Nachrichtenanalyst",
        goal=(
            "Ueberwache, filtere und fasse die relevantesten Finanznachrichten "
            "fuer bestimmte Aktien zusammen. Bewerte die potenzielle Auswirkung jeder Nachricht "
            "auf die Kursentwicklung und liefere umsetzbare Erkenntnisse."
        ),
        backstory=(
            "Du bist ein erfahrener Finanzjournalist, der zum Analysten wurde, mit tiefen "
            "Verbindungen in die Medienlandschaft. Du verfolgst Breaking News, Quartalsberichte, "
            "regulatorische Aenderungen und Marktereignisse, die Aktienkurse bewegen. "
            "Du zeichnest dich darin aus, Signal von Rauschen zu trennen, wesentliche "
            "Informationen zu identifizieren und einzuschaetzen, wie Nachrichten die "
            "Anlegerstimmung und Aktienbewertungen beeinflussen. "
            "Du antwortest immer auf Deutsch."
        ),
        tool_group="news",
        model=model or DEFAULT_MODEL,
    )


def create_portfolio_monitor_agent(model: str | None = None) -> Agent:
    """Portfolio Manager – holdings tracking and portfolio-level insights."""
    return Agent(
        role="Portfolio Manager",
        goal=(
            "Ueberwache das aktuelle Portfolio, berechne Performance-Kennzahlen, "
            "identifiziere Top- und Underperformer, bewerte das Gesamtrisiko "
            "und gib strategische Empfehlungen zur Portfolio-Optimierung."
        ),
        backstory=(
            "Du bist ein zertifizierter Portfolio-Manager mit Expertise in Risikomanagement "
            "und Asset Allocation. Du ueberwachst Positionen in Echtzeit, verfolgst Einstandskurse "
            "und Renditen, identifizierst Konzentrationsrisiken und schlaegst Rebalancing-Strategien "
            "vor. Du kommunizierst die Portfolio-Gesundheit in klaren, praegnanten Berichten, "
            "die Anlegern helfen, fundierte Entscheidungen zu treffen. "
            "Du antwortest immer auf Deutsch."
        ),
        tool_group="portfolio",
        model=model or DEFAULT_MODEL,
    )
