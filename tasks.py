"""
Task definitions for the multi-agent system.

Each function returns a task description string that is passed to an agent.
Tasks define WHAT the agent should do – the agent decides HOW (which tools to use).
This separation of concerns mirrors professional agent frameworks.

All task descriptions are in German so agents respond in German.
"""


def stock_analysis_task(ticker: str) -> str:
    """Task for the Research Agent: comprehensive fundamental analysis."""
    return (
        f"Fuehre eine umfassende Fundamentalanalyse von {ticker} durch.\n\n"
        "Schritte:\n"
        f"1. Nutze get_stock_data, um Finanzkennzahlen fuer {ticker} abzurufen.\n"
        f"2. Nutze get_company_info, um Unternehmensdetails fuer {ticker} zu erhalten.\n"
        "3. Fasse alle Daten in einem professionellen Investment-Bericht zusammen.\n\n"
        "Dein Bericht MUSS folgende Abschnitte enthalten:\n"
        "## Unternehmensueberblick\n"
        "Was das Unternehmen macht, Sektor, Branche, wichtige Fakten.\n\n"
        "## Finanzkennzahlen\n"
        "Kurs, Marktkapitalisierung, KGV, Umsatz, Margen, EPS – alle Schluesselzahlen.\n\n"
        "## Investment-These\n"
        "Warum diese Aktie attraktiv ist oder nicht. Belege mit konkreten Daten.\n\n"
        "## Wachstumskatalysatoren\n"
        "Wesentliche Treiber, die kuenftiges Wachstum antreiben koennten.\n\n"
        "## Risikobewertung\n"
        "Groesste Risiken, Bedenken und was schiefgehen koennte.\n\n"
        "## Empfehlung\n"
        "Klare **KAUFEN**, **HALTEN** oder **VERKAUFEN** Empfehlung mit Begruendung.\n"
        "\nAntworte auf Deutsch.\n"
    )


def news_analysis_task(ticker: str) -> str:
    """Task for the News Agent: fetch and summarize recent news."""
    return (
        f"Analysiere aktuelle Nachrichten zu {ticker}.\n\n"
        f"1. Nutze search_news, um aktuelle Artikel zu {ticker} abzurufen.\n"
        "2. Fuer jeden Artikel liefere:\n"
        "   - **Schlagzeile**: Originaltitel\n"
        "   - **Zusammenfassung**: 2-3 Saetze\n"
        "   - **Auswirkung**: \U0001f7e2 POSITIV, \U0001f7e1 NEUTRAL oder \U0001f534 NEGATIV\n"
        "   - **Begruendung**: Warum du diese Auswirkung so einschaetzt\n\n"
        "3. Schliesse mit einem Abschnitt **Gesamtstimmung** ab, der die kollektive Auswirkung zusammenfasst.\n"
        "4. Falls keine Nachrichten gefunden werden, sage das und liefere allgemeinen Marktkontext.\n"
        "\nAntworte auf Deutsch.\n"
    )


def portfolio_analysis_task() -> str:
    """Task for the Portfolio Monitor Agent: analyze the full portfolio."""
    return (
        "Analysiere das aktuelle Investmentportfolio.\n\n"
        "1. Nutze get_portfolio_data, um alle Positionen zu lesen.\n"
        "2. Nutze calculate_returns, um Live-Performance-Daten zu erhalten.\n"
        "3. Erstelle einen umfassenden Portfolio-Gesundheitsbericht.\n\n"
        "Dein Bericht MUSS enthalten:\n"
        "## Portfolio-Zusammenfassung\n"
        "Gesamtwert, investiertes Kapital, Gesamtrendite in Prozent.\n\n"
        "## Positionsuebersicht\n"
        "Performance jeder einzelnen Position mit konkreten Zahlen.\n\n"
        "## Top-Performer\n"
        "Beste Aktie mit Analyse, warum sie so gut laeuft.\n\n"
        "## Schlechtester Performer\n"
        "Schwaechste Aktie mit Analyse und ob halten oder verkaufen.\n\n"
        "## Risikobewertung\n"
        "Konzentrationsrisiko, Branchenexposition, Diversifikation.\n\n"
        "## Empfehlungen\n"
        "Konkrete, umsetzbare Schritte zur Verbesserung der Portfolio-Performance.\n"
        "\nAntworte auf Deutsch.\n"
    )
