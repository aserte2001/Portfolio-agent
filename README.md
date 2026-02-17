# 🤖 AI Portfolio Agent

> Eine produktionsreife Multi-Agent-KI-Anwendung, bei der 3 spezialisierte KI-Agenten zusammenarbeiten, um Aktien zu analysieren und ein Investmentportfolio zu verwalten. Gebaut mit einem eigenen leichtgewichtigen Agent-Framework auf Basis von OpenAI Function Calling.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai)
![Agents](https://img.shields.io/badge/Architektur-Multi--Agent-purple)

---

## 🎯 Demonstrierte Skills

| Skill | Umsetzung |
|-------|-----------|
| 🤖 **Multi-Agent-Systeme** | 3 spezialisierte Agenten mit eigenen Rollen, Zielen und Tools |
| 🧠 **LLM-Orchestrierung** | OpenAI GPT-4o mit autonomer Function-Calling-Schleife |
| 🔧 **Tool-Integration** | Eigene Python-Tools, die die yfinance-API wrappen |
| 💬 **Konversationelle KI** | Chat mit Intent-Klassifikation und Agent-Routing |
| 🧠 **Memory-System** | Persistentes Anlageprofil, das in alle Agenten injiziert wird |
| 🎯 **Prompt Engineering** | Optimierte System-Prompts, Agent-Backstories und Task-Beschreibungen |
| 🏗️ **Produktionsmuster** | Fehlerbehandlung, Logging, Konfigurationsmanagement, Session State |
| 🎨 **Modernes UI/UX** | Streamlit mit Custom CSS, Gradient-Cards, Fortschrittsanzeigen |

---

## 🧠 Agenten-Architektur

```
┌──────────────────────────────────────────────────────────────────┐
│                   Eigenes Agent-Framework                        │
│              (OpenAI Function Calling Loop)                      │
├───────────────────┬───────────────────┬─────────────────────────┤
│  🔍 Research      │  📰 News          │  💼 Portfolio Monitor    │
│  Agent            │  Agent            │  Agent                  │
├───────────────────┼───────────────────┼─────────────────────────┤
│ Rolle: Senior     │ Rolle: Finanz-    │ Rolle: Portfolio        │
│ Investment        │ Nachrichtenanalyst│ Manager                 │
│ Analyst           │                   │                         │
├───────────────────┼───────────────────┼─────────────────────────┤
│ Tools:            │ Tools:            │ Tools:                  │
│ • get_stock_data  │ • search_news     │ • get_portfolio_data    │
│ • get_company_    │                   │ • calculate_returns     │
│   info            │                   │                         │
├───────────────────┼───────────────────┼─────────────────────────┤
│ Output:           │ Output:           │ Output:                 │
│ Investment-       │ News-Briefing     │ Portfolio-              │
│ These + Empf.     │ + Auswirkung      │ Gesundheitsbericht      │
└───────────────────┴───────────────────┴─────────────────────────┘
```

### So funktioniert die Agentic Loop

```python
# Vereinfachte Version der Kern-Agent-Schleife (siehe agents.py)

while iterations < max_iterations:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=agent.tools,      # Function-Schemas
        tool_choice="auto",     # LLM entscheidet, wann Tools genutzt werden
    )

    if response.tool_calls:
        # Agent entscheidet autonom, ein Tool aufzurufen
        for call in response.tool_calls:
            result = execute_tool(call.function.name, call.function.arguments)
            messages.append(tool_result)  # Ergebnis zurueckgeben
        continue  # LLM verarbeitet die Tool-Ergebnisse

    return response.content  # Agent ist fertig – finale Analyse zurueckgeben
```

---

## 🚀 Schnellstart

### 1. Klonen & Installieren

```bash
git clone https://github.com/deinbenutzername/ai-portfolio-agent.git
cd ai-portfolio-agent
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. API Key konfigurieren

```bash
cp .env.example .env
# .env bearbeiten und deinen OpenAI API Key eintragen
```

Oder direkt in der App unter **Einstellungen** konfigurieren.

### 3. Starten

```bash
streamlit run app.py
```

Oeffnet automatisch unter `http://localhost:8501`.

---

## 📁 Projektstruktur

```
ai-portfolio-agent/
├── app.py                    # Streamlit-Webanwendung (6 Seiten, Custom CSS)
├── agents.py                 # Agent-Framework + 3 Agenten-Definitionen
├── tasks.py                  # Task-Beschreibungen fuer jeden Agenten
├── crew.py                   # Workflow-Orchestrierung (Single- + Multi-Agent)
├── tools.py                  # Tool-Funktionen + OpenAI-Schemas + Portfolio-Verwaltung
├── chat_handler.py           # Chat-Routing, Intent-Klassifikation, Verlauf
├── memory_manager.py         # Anlageprofil laden/speichern/injizieren
├── config.py                 # Konfiguration und Umgebungsvariablen
├── portfolio.json            # Portfolio-Daten (vorausgefuellt mit Beispielen)
├── investment_profile.json   # Nutzer-Anlageprofil (Memory-System)
├── chat_history.json         # Chat-Verlauf
├── requirements.txt          # Python-Dependencies (minimal: 4 Pakete)
├── .env.example              # API-Key-Vorlage
├── .gitignore                # Git Ignore Rules
└── README.md                 # Diese Datei
```

---

## 🔧 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) – Modernes Python-Web-UI
- **LLM:** [OpenAI GPT-4o](https://openai.com/) – Function Calling fuer Tool-Nutzung
- **Finanzdaten:** [yfinance](https://github.com/ranaroussi/yfinance) – Yahoo Finance API
- **Architektur:** Eigenes Multi-Agent-Framework (gleiche Muster wie CrewAI/LangGraph)

---

## 📝 Lizenz

MIT-Lizenz – gerne als Portfolio-Stueck oder Lernressource verwenden.
