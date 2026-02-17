"""
AI Portfolio Agent – Tile-Based Dashboard
Modern Fintech Kachel-Design (Notion / Linear / Trade Republic)
"""

import logging
import os
import time
from datetime import datetime

import streamlit as st
import yfinance as yf

from config import APP_TITLE, APP_ICON, SUPPORTED_MODELS, DEFAULT_MODEL
from tools import load_portfolio, save_portfolio, add_to_portfolio, remove_from_portfolio
from crew import run_stock_analysis, run_news_analysis, run_portfolio_analysis
from memory_manager import (
    load_profile, save_profile, reset_profile, is_profile_configured,
    get_profile_summary, SECTOR_OPTIONS, RISK_LEVELS, HORIZONS,
    STOCK_SIZES, GEO_OPTIONS,
)
from chat_handler import (
    load_chat_history, save_chat_history, clear_chat_history,
    add_message, handle_chat_message, AGENT_META,
)
from currency_converter import (
    usd_to_eur, format_eur, get_usd_to_eur_rate, is_using_fallback,
    to_eur, detect_currency,
)

# ═══════════════════════════════════════════════════════════════════
# LOGGING & PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Portfolio Agent",
    page_icon="\U0001f4bc",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# LOAD CSS
# ═══════════════════════════════════════════════════════════════════

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "news_result" not in st.session_state:
    st.session_state.news_result = None
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_chat_history()

# ═══════════════════════════════════════════════════════════════════
# TILE HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_price_raw(ticker: str) -> tuple[float | None, str]:
    """Fetch current price with multiple fallbacks.
    Returns (price, native_currency) tuple.
    The native_currency is auto-detected from yfinance info."""
    native_currency = "USD"
    try:
        stock = yf.Ticker(ticker)

        # Try to detect native currency early
        try:
            info = stock.info or {}
            native_currency = detect_currency(info)
        except Exception:
            pass

        try:
            price = getattr(stock.fast_info, "last_price", None)
            if price is not None and price > 0:
                logger.info(f"[Price] {ticker}: {price:.2f} {native_currency} (fast_info)")
                return round(price, 2), native_currency
        except Exception:
            pass

        try:
            info = stock.info or {}
            native_currency = detect_currency(info)
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if price is not None and price > 0:
                logger.info(f"[Price] {ticker}: {price:.2f} {native_currency} (info)")
                return round(price, 2), native_currency
        except Exception:
            pass

        try:
            hist = stock.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].dropna().iloc[-1])
                if price > 0:
                    logger.info(f"[Price] {ticker}: {price:.2f} {native_currency} (history)")
                    return round(price, 2), native_currency
        except Exception:
            pass

        logger.warning(f"[Price] {ticker}: all fallbacks failed")
        return None, native_currency

    except Exception as e:
        logger.error(f"[Price] {ticker}: exception {e}")
        return None, native_currency


def get_current_price_eur(ticker: str) -> float | None:
    """Fetch current price and convert to EUR if needed.
    Automatically detects if the ticker already trades in EUR
    and skips conversion in that case."""
    price, native_currency = get_price_raw(ticker)
    if price is None:
        return None
    eur = to_eur(price, native_currency)
    return round(eur, 2) if eur is not None else None


def tile(icon: str, title: str, content: str, glow: str = "", extra_class: str = "") -> str:
    """Render a dashboard tile with icon header and HTML content."""
    glow_class = f"tile-glow-{glow}" if glow else ""
    return (
        f'<div class="tile {glow_class} {extra_class}">'
        f'<div class="tile-header">'
        f'<span class="tile-icon">{icon}</span>'
        f'<span class="tile-title">{title}</span>'
        f'</div>'
        f'{content}'
        f'</div>'
    )


def metric_tile(icon: str, title: str, value: str, color: str = "", sub: str = "") -> str:
    """Render a metric tile with big number."""
    glow = color if color else ""
    sub_html = f'<div class="tile-sub">{sub}</div>' if sub else ""
    content = (
        f'<div class="tile-label">{title}</div>'
        f'<div class="tile-value {color}">{value}</div>'
        f'{sub_html}'
    )
    return (
        f'<div class="tile tile-glow-{glow}" style="text-align:center;">'
        f'<div style="font-size:28px;margin-bottom:10px;">{icon}</div>'
        f'{content}'
        f'</div>'
    )


def holding_tile_html(ticker: str, shares: float, cost_basis_eur: float,
                      current_price_eur: float | None) -> str:
    """Render a single holding as a card tile (all values already in EUR)."""
    if current_price_eur is not None:
        current_val = shares * current_price_eur
        invested = shares * cost_basis_eur
        gain = current_val - invested
        ret_pct = ((current_price_eur - cost_basis_eur) / cost_basis_eur) * 100 if cost_basis_eur > 0 else 0
        ret_class = "pos" if ret_pct >= 0 else "neg"
        sign = "+" if ret_pct >= 0 else ""
        gain_sign = "+" if gain >= 0 else ""
        return (
            f'<div class="holding-tile">'
            f'<div class="ticker">{ticker}</div>'
            f'<div class="shares">{shares:.0f} Anteile</div>'
            f'<div class="price-row">'
            f'<span class="current-val">{format_eur(current_val)}</span>'
            f'<span class="return-pct {ret_class}">{sign}{ret_pct:.1f}%</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'<span>Einstand: {format_eur(cost_basis_eur)}</span>'
            f'<span>Aktuell: {format_eur(current_price_eur)}</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'<span>Investiert: {format_eur(invested)}</span>'
            f'<span>{gain_sign}{format_eur(abs(gain))}</span>'
            f'</div>'
            f'</div>'
        )
    return (
        f'<div class="holding-tile">'
        f'<div class="ticker">{ticker}</div>'
        f'<div class="shares">{shares:.0f} Anteile @ {format_eur(cost_basis_eur)}</div>'
        f'<div class="price-row">'
        f'<span class="current-val" style="color:#8B93B0;">N/A</span>'
        f'</div>'
        f'</div>'
    )


def check_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key != "your_key_here")


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# \U0001f4bc Portfolio Agent")
    st.markdown("---")

    portfolio_count = len(load_portfolio())

    page = st.radio(
        "Navigation",
        [
            "\U0001f4ca Aktie analysieren",
            f"\U0001f4bc Mein Portfolio ({portfolio_count})",
            "\U0001f4f0 News Feed",
            "\U0001f4ac Chat",
            "\U0001f9e0 Anlageprofil",
            "\u2699\ufe0f Einstellungen",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if is_profile_configured():
        summary = get_profile_summary()
        st.markdown(f'<div class="profile-badge">\U0001f9e0 {summary}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="profile-badge">\U0001f9e0 Profil nicht eingerichtet<br>'
            '<small>Gehe zu Anlageprofil</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Exchange rate display
    rate = get_usd_to_eur_rate()
    rate_label = "\u26a0\ufe0f geschaetzt" if is_using_fallback() else "live"
    st.markdown(
        f'<div style="font-size:12px;color:#4A5270;margin-top:4px;">'
        f'\U0001f4b1 1 USD = {rate:.4f} EUR <small>({rate_label})</small>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:12px;color:#4A5270;margin-top:8px;">'
        '<span class="status-dot live"></span> 3 Agenten bereit<br>'
        '\U0001f50d Research &bull; \U0001f4f0 News &bull; \U0001f4ca Portfolio'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# SEITE 1: AKTIE ANALYSIEREN
# ═══════════════════════════════════════════════════════════════════

if page == "\U0001f4ca Aktie analysieren":

    st.markdown(
        '<div class="page-header">'
        "<h1>\U0001f4ca Aktienanalyse</h1>"
        "<p>KI-gestuetzte Fundamentalanalyse mit dem Research Agent</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Search tile (full width)
    st.markdown(
        tile("\U0001f50d", "Aktie suchen",
             '<div class="tile-sub">Gib ein Ticker-Symbol ein und starte die KI-Analyse</div>'),
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker", placeholder="z.B. RKLB, AAPL, TSLA",
            label_visibility="collapsed",
        ).upper().strip()
    with col2:
        analyze_btn = st.button("\U0001f50d Analyse starten", type="primary", use_container_width=True)
    with col3:
        if st.button("\U0001f5d1\ufe0f Leeren", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.last_ticker = ""
            st.rerun()

    # Quick-access tiles for portfolio tickers
    portfolio = load_portfolio()
    if portfolio and not st.session_state.analysis_result:
        ptickers = [h["ticker"] for h in portfolio]
        st.markdown("### SCHNELLANALYSE")
        qcols = st.columns(min(len(ptickers), 4))
        for i, t in enumerate(ptickers[:4]):
            with qcols[i]:
                if st.button(f"\U0001f4ca {t}", key=f"qa_{t}", use_container_width=True):
                    ticker_input = t
                    analyze_btn = True

    # Run analysis
    if analyze_btn and ticker_input:
        if not check_api_key():
            st.error("\u26a0\ufe0f OpenAI API Key nicht konfiguriert. Gehe zu **Einstellungen**.")
        else:
            st.session_state.last_ticker = ticker_input
            with st.status(f"\U0001f916 Analysiere {ticker_input}...", expanded=True) as status:
                st.write("\U0001f50d Research Agent sammelt Finanzdaten...")
                st.write("\U0001f4ca Rufe Kennzahlen und Unternehmensinformationen ab...")
                try:
                    result = run_stock_analysis(ticker_input, model=st.session_state.selected_model)
                    st.session_state.analysis_result = result
                    status.update(label=f"\u2705 Analyse abgeschlossen fuer {ticker_input}", state="complete")
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    st.error(f"\u274c Analyse fehlgeschlagen: {str(e)}")
                    status.update(label="\u274c Analyse fehlgeschlagen", state="error")
    elif analyze_btn and not ticker_input:
        st.warning("\u26a0\ufe0f Bitte gib ein Aktien-Tickersymbol ein.")

    # Results as tiles
    if st.session_state.analysis_result:
        td = st.session_state.last_ticker

        # Metric tiles row
        try:
            qi = yf.Ticker(td).info
            native_cur = detect_currency(qi)
            q_price_raw = qi.get("regularMarketPrice") or qi.get("currentPrice")
            q_mcap_raw = qi.get("marketCap", 0)
            q_pe = qi.get("trailingPE", "N/A")
            q_sector = qi.get("sector", "N/A")
            q_52h_raw = qi.get("fiftyTwoWeekHigh")
            q_52l_raw = qi.get("fiftyTwoWeekLow")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    metric_tile("\U0001f4b0", "AKTUELLER KURS",
                                format_eur(to_eur(q_price_raw, native_cur)) if q_price_raw else "N/A", "blue"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_tile("\U0001f3e6", "MARKTKAPITALISIERUNG",
                                format_eur(to_eur(q_mcap_raw, native_cur)) if q_mcap_raw else "N/A", "purple"),
                    unsafe_allow_html=True,
                )
            with c3:
                pe_val = f"{q_pe:.1f}" if isinstance(q_pe, (int, float)) else "N/A"
                st.markdown(
                    metric_tile("\U0001f4ca", "KGV", pe_val),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_tile("\U0001f3af", "SEKTOR", q_sector),
                    unsafe_allow_html=True,
                )

            # 52w range tiles
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    metric_tile("\U0001f4c9", "52W TIEF",
                                format_eur(to_eur(q_52l_raw, native_cur)) if q_52l_raw else "N/A", "red"),
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    metric_tile("\U0001f4c8", "52W HOCH",
                                format_eur(to_eur(q_52h_raw, native_cur)) if q_52h_raw else "N/A", "green"),
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # Analysis report tile
        with st.expander(f"\U0001f4cb Vollstaendiger Analysebericht – {td}", expanded=True):
            st.markdown(st.session_state.analysis_result)

        # Add to portfolio tile
        st.markdown(
            tile("\U0001f4bc", f"{td} ins Portfolio aufnehmen",
                 '<div class="tile-sub">Lege Anteile und Kaufpreis in Euro fest</div>'),
            unsafe_allow_html=True,
        )
        cp_now = get_current_price_eur(td)
        if cp_now:
            st.markdown(
                f'<div style="color:#8B93B0;font-size:13px;margin-bottom:8px;">'
                f'Aktueller Marktpreis: <b style="color:#5B7FFF;">{format_eur(cp_now)}</b></div>',
                unsafe_allow_html=True,
            )
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            add_shares = st.number_input("Anteile", min_value=0.01, value=10.0, step=1.0)
        with pc2:
            add_cost = st.number_input("Dein Kaufpreis pro Aktie (\u20ac)", min_value=0.01,
                                       value=0.01, step=0.01,
                                       help="Preis den du pro Aktie bezahlt hast in Euro")
        with pc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"\u2795 {td} hinzufuegen", use_container_width=True):
                if add_cost <= 0.01:
                    st.warning("Bitte gib deinen echten Kaufpreis in \u20ac ein.")
                else:
                    msg = add_to_portfolio(td, add_shares, add_cost)
                    st.success(msg)


# ═══════════════════════════════════════════════════════════════════
# SEITE 2: PORTFOLIO DASHBOARD
# ═══════════════════════════════════════════════════════════════════

elif page.startswith("\U0001f4bc Mein Portfolio"):

    st.markdown(
        '<div class="page-header">'
        "<h1>\U0001f4bc Portfolio Dashboard</h1>"
        "<p>Live-Tracking deiner Positionen mit KI-gestuetzten Einblicken</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    portfolio = load_portfolio()

    if not portfolio:
        st.markdown(
            tile("\U0001f4ed", "Portfolio leer",
                 '<div class="tile-sub">Gehe zu <b>Aktie analysieren</b>, um deine erste Position hinzuzufuegen.</div>',
                 glow="blue"),
            unsafe_allow_html=True,
        )
    else:
        total_invested = 0.0
        total_current = 0.0
        holdings = []
        best = {"ticker": "", "return_pct": -999}
        worst = {"ticker": "", "return_pct": 999}

        progress = st.progress(0, text="Lade aktuelle Kurse...")
        for i, h in enumerate(portfolio):
            t, s = h["ticker"], h["shares"]
            cb_eur = h.get("cost_basis_eur", h.get("cost_basis", 0))
            ct = s * cb_eur
            cp = get_current_price_eur(t)

            if cp is not None:
                cv = s * cp
                gl = cv - ct
                rp = ((cp - cb_eur) / cb_eur) * 100 if cb_eur > 0 else 0
                total_invested += ct
                total_current += cv
                if rp > best["return_pct"]:
                    best = {"ticker": t, "return_pct": rp}
                if rp < worst["return_pct"]:
                    worst = {"ticker": t, "return_pct": rp}
            else:
                cv = ct
                gl = 0
                rp = 0
                cp = None

            holdings.append({"ticker": t, "shares": s, "cost_basis_eur": cb_eur,
                             "current_price_eur": cp, "value": cv,
                             "gain": gl, "return_pct": rp})
            progress.progress((i + 1) / len(portfolio), text=f"{t} geladen...")
        progress.empty()

        total_gain = total_current - total_invested
        total_return_pct = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0
        gain_color = "green" if total_gain >= 0 else "red"
        sign = "+" if total_gain >= 0 else ""

        # ── Top Metric Tiles ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                metric_tile("\U0001f4b0", "PORTFOLIO-WERT",
                            format_eur(total_current), "blue"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_tile("\U0001f4c8", "GEWINN / VERLUST",
                            f"{sign}{format_eur(abs(total_gain))}", gain_color),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_tile("\U0001f4ca", "RENDITE",
                            f"{sign}{total_return_pct:.1f}%", gain_color),
                unsafe_allow_html=True,
            )
        with c4:
            bp = best["ticker"] or "\u2013"
            bpr = f" ({best['return_pct']:+.1f}%)" if best["ticker"] else ""
            st.markdown(
                metric_tile("\U0001f3c6", "TOP PERFORMER",
                            f"{bp}{bpr}", "green"),
                unsafe_allow_html=True,
            )

        # ── Holdings Grid (Kacheln!) ──
        st.markdown("### MEINE POSITIONEN")

        num_cols = min(len(holdings), 4)
        if num_cols > 0:
            rows = (len(holdings) + num_cols - 1) // num_cols
            for row in range(rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    h_idx = row * num_cols + col_idx
                    if h_idx < len(holdings):
                        h = holdings[h_idx]
                        with cols[col_idx]:
                            st.markdown(
                                holding_tile_html(
                                    h["ticker"], h["shares"],
                                    h["cost_basis_eur"], h["current_price_eur"],
                                ),
                                unsafe_allow_html=True,
                            )

        # ── AI Analysis Tile ──
        st.markdown(
            tile("\U0001f916", "KI-Portfolio-Analyse",
                 '<div class="tile-sub">Der Portfolio Monitor Agent analysiert Diversifikation, Risiken und gibt Empfehlungen</div>',
                 glow="purple"),
            unsafe_allow_html=True,
        )

        if st.button("\U0001f4ca Analyse starten", type="primary", use_container_width=True):
            if not check_api_key():
                st.error("\u26a0\ufe0f OpenAI API Key nicht konfiguriert.")
            else:
                with st.status("\U0001f916 Portfolio Monitor analysiert...", expanded=True) as status:
                    st.write("\U0001f4bc Lese Portfolio-Positionen...")
                    st.write("\U0001f4c8 Berechne Renditen und Risikokennzahlen...")
                    try:
                        result = run_portfolio_analysis(model=st.session_state.selected_model)
                        st.session_state.portfolio_result = result
                        status.update(label="\u2705 Portfolio-Analyse abgeschlossen", state="complete")
                    except Exception as e:
                        logger.error(f"Portfolio analysis failed: {e}")
                        st.error(f"\u274c Analyse fehlgeschlagen: {str(e)}")
                        status.update(label="\u274c Analyse fehlgeschlagen", state="error")

        if st.session_state.portfolio_result:
            with st.expander("\U0001f4ca KI-Portfolio-Analyse", expanded=True):
                st.markdown(st.session_state.portfolio_result)

        # ── Manage Tiles ──
        st.markdown("### VERWALTEN")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                tile("\u2795", "Aktie hinzufuegen",
                     '<div class="tile-sub">Neue Position ins Portfolio aufnehmen</div>'),
                unsafe_allow_html=True,
            )
            nt = st.text_input("Ticker", placeholder="z.B. AAPL", key="new_ticker").upper().strip()
            ns = st.number_input("Anteile", min_value=0.01, value=10.0, step=1.0, key="new_shares")
            nc = st.number_input("Dein Kaufpreis (\u20ac)", min_value=0.01, value=0.01, step=0.01, key="new_cost",
                                help="Preis den du pro Aktie bezahlt hast in Euro")
            if st.button("\u2795 Hinzufuegen", key="add_btn", use_container_width=True):
                if not nt:
                    st.warning("Bitte gib einen Ticker ein.")
                elif nc <= 0.01:
                    st.warning("Bitte gib deinen echten Kaufpreis in \u20ac ein.")
                else:
                    st.success(add_to_portfolio(nt, ns, nc))
                    time.sleep(0.5)
                    st.rerun()
        with m2:
            st.markdown(
                tile("\U0001f5d1\ufe0f", "Aktie entfernen",
                     '<div class="tile-sub">Position aus dem Portfolio loeschen</div>'),
                unsafe_allow_html=True,
            )
            tickers = [h["ticker"] for h in portfolio]
            if tickers:
                rt = st.selectbox("Ticker waehlen", tickers)
                if st.button("\U0001f5d1\ufe0f Entfernen", key="rem_btn", use_container_width=True):
                    st.success(remove_from_portfolio(rt))
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.info("Keine Aktien vorhanden.")


# ═══════════════════════════════════════════════════════════════════
# SEITE 3: NEWS FEED
# ═══════════════════════════════════════════════════════════════════

elif page == "\U0001f4f0 News Feed":

    st.markdown(
        '<div class="page-header">'
        "<h1>\U0001f4f0 KI-News Feed</h1>"
        "<p>Kursrelevante Nachrichten zusammengefasst und bewertet vom News Agent</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Search tile
    st.markdown(
        tile("\U0001f4f0", "News-Suche",
             '<div class="tile-sub">Ticker eingeben oder aus deinem Portfolio waehlen</div>'),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        news_ticker = st.text_input("Ticker", placeholder="z.B. RKLB, AAPL",
                                    label_visibility="collapsed").upper().strip()
    with col2:
        news_btn = st.button("\U0001f4f0 News abrufen", type="primary", use_container_width=True)

    # Portfolio quick-select as tiles
    portfolio = load_portfolio()
    ptickers = [h["ticker"] for h in portfolio]
    if ptickers:
        st.markdown("### DEIN PORTFOLIO")
        ncols = st.columns(min(len(ptickers), 4))
        for i, t in enumerate(ptickers[:4]):
            with ncols[i]:
                tile_html = (
                    f'<div class="tile tile-glow-blue" style="text-align:center;cursor:pointer;">'
                    f'<div style="font-size:24px;margin-bottom:6px;">\U0001f4f0</div>'
                    f'<div class="tile-value blue" style="font-size:20px;">{t}</div>'
                    f'<div class="tile-sub">News abrufen</div>'
                    f'</div>'
                )
                st.markdown(tile_html, unsafe_allow_html=True)
                if st.button(f"News {t}", key=f"nq_{t}", use_container_width=True):
                    news_ticker = t
                    news_btn = True

    if news_btn and news_ticker:
        if not check_api_key():
            st.error("\u26a0\ufe0f OpenAI API Key nicht konfiguriert.")
        else:
            with st.status(f"\U0001f916 News Agent analysiert {news_ticker}...", expanded=True) as status:
                st.write(f"\U0001f4f0 Rufe Nachrichten fuer {news_ticker} ab...")
                st.write("\U0001f9e0 KI fasst zusammen und bewertet Auswirkungen...")
                try:
                    result = run_news_analysis(news_ticker, model=st.session_state.selected_model)
                    st.session_state.news_result = result
                    st.session_state.news_ticker = news_ticker
                    status.update(label=f"\u2705 News-Analyse fuer {news_ticker}", state="complete")
                except Exception as e:
                    logger.error(f"News analysis failed: {e}")
                    st.error(f"\u274c Fehlgeschlagen: {str(e)}")
                    status.update(label="\u274c Fehler", state="error")
    elif news_btn and not news_ticker:
        st.warning("\u26a0\ufe0f Bitte gib einen Ticker ein.")

    if st.session_state.news_result:
        with st.expander(f"\U0001f4f0 KI-News – {st.session_state.get('news_ticker', '')}", expanded=True):
            st.markdown(st.session_state.news_result)


# ═══════════════════════════════════════════════════════════════════
# SEITE 4: CHAT
# ═══════════════════════════════════════════════════════════════════

elif page == "\U0001f4ac Chat":

    st.markdown(
        '<div class="page-header">'
        "<h1>\U0001f4ac Chat mit deinen Agenten</h1>"
        "<p>Die KI leitet deine Fragen automatisch an den passenden Spezialagenten</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Two-column layout: Chat + Context panel
    chat_col, ctx_col = st.columns([3, 1])

    with ctx_col:
        # Context panel
        ctx_parts = []
        if is_profile_configured():
            ctx_parts.append(
                '<div class="ctx-label">PROFIL</div>'
                f'<div class="ctx-value">\U0001f9e0 {get_profile_summary()}</div>'
            )
        else:
            ctx_parts.append(
                '<div class="ctx-label">PROFIL</div>'
                '<div class="ctx-value" style="color:#6B7394;">Nicht konfiguriert</div>'
            )

        portfolio = load_portfolio()
        ctx_parts.append(
            '<div class="ctx-label">PORTFOLIO</div>'
            f'<div class="ctx-value">\U0001f4bc {len(portfolio)} Positionen</div>'
        )

        ctx_parts.append(
            '<div class="ctx-label">AGENTEN</div>'
            '<div class="ctx-value">'
            '<span class="status-dot live"></span> Research<br>'
            '<span class="status-dot live"></span> News<br>'
            '<span class="status-dot live"></span> Portfolio'
            '</div>'
        )

        ctx_parts.append(
            '<div class="ctx-label">MODELL</div>'
            f'<div class="ctx-value">{st.session_state.selected_model}</div>'
        )

        st.markdown(
            f'<div class="context-tile">{"".join(ctx_parts)}</div>',
            unsafe_allow_html=True,
        )

        if st.button("\U0001f5d1\ufe0f Chat leeren", use_container_width=True):
            clear_chat_history()
            st.session_state.chat_messages = []
            st.rerun()

    with chat_col:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="\U0001f464"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                agent_type = msg.get("agent_type", "research")
                meta = AGENT_META.get(agent_type, AGENT_META["research"])
                with st.chat_message("assistant", avatar=meta["emoji"]):
                    st.markdown(
                        f'<span class="agent-badge" style="background:{meta["color"]};">'
                        f'{meta["emoji"]} {meta["name"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Frag deine Agenten... (z.B. 'Soll ich RKLB kaufen?')"):
            if not check_api_key():
                st.error("\u26a0\ufe0f OpenAI API Key nicht konfiguriert.")
            else:
                st.session_state.chat_messages = add_message(
                    st.session_state.chat_messages, "user", prompt
                )
                with st.chat_message("user", avatar="\U0001f464"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="\U0001f916"):
                    with st.spinner("\U0001f916 Leite an den passenden Agenten..."):
                        response, agent_type = handle_chat_message(
                            prompt,
                            st.session_state.chat_messages,
                            model=st.session_state.selected_model,
                        )
                    meta = AGENT_META.get(agent_type, AGENT_META["research"])
                    st.markdown(
                        f'<span class="agent-badge" style="background:{meta["color"]};">'
                        f'{meta["emoji"]} {meta["name"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(response)

                st.session_state.chat_messages = add_message(
                    st.session_state.chat_messages, "assistant", response, agent_type
                )
                save_chat_history(st.session_state.chat_messages)

        if not st.session_state.chat_messages:
            st.markdown("### PROBIERE ZUM BEISPIEL")
            examples = [
                "\U0001f50d Soll ich RKLB bei \u20ac7,82 kaufen?",
                "\u26a0\ufe0f Was sind die Risiken in meinem Portfolio?",
                "\U0001f4f0 Gibt es aktuelle News zu AAPL?",
                "\U0001f4ca Ist mein Portfolio zu konzentriert?",
                "\U0001f4a1 Was ist die Investment-These fuer NVDA?",
                "\U0001f4c8 Wie hat sich mein Portfolio entwickelt?",
            ]
            ex_cols = st.columns(2)
            for i, ex in enumerate(examples):
                with ex_cols[i % 2]:
                    st.markdown(
                        f'<div class="tile" style="padding:14px 18px;margin-bottom:8px;">'
                        f'<span style="color:#8B93B0;font-size:13px;">{ex}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════
# SEITE 5: ANLAGEPROFIL
# ═══════════════════════════════════════════════════════════════════

elif page == "\U0001f9e0 Anlageprofil":

    st.markdown(
        '<div class="page-header">'
        "<h1>\U0001f9e0 Mein Anlageprofil</h1>"
        "<p>Bringe deinen Agenten deinen Anlagestil bei – sie merken sich alles</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    profile = load_profile()

    if is_profile_configured():
        st.success(f"\u2705 Profil aktiv: **{get_profile_summary()}**")
        if profile.get("last_updated"):
            try:
                dt = datetime.fromisoformat(profile["last_updated"])
                st.caption(f"Zuletzt aktualisiert: {dt.strftime('%d.%m.%Y um %H:%M')}")
            except (ValueError, TypeError):
                pass

        # Profile tiles
        risk_emoji = {"Niedrig": "\U0001f6e1\ufe0f", "Mittel": "\u2696\ufe0f",
                      "Hoch": "\u26a1", "Sehr hoch": "\U0001f525"}
        re = risk_emoji.get(profile.get("risk_tolerance", ""), "\u2696\ufe0f")

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(
                metric_tile(re, "RISIKOBEREITSCHAFT",
                            profile.get("risk_tolerance", "N/A")),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_tile("\U0001f3af", "ANLAGEHORIZONT",
                            profile.get("investment_horizon", "N/A"), "blue"),
                unsafe_allow_html=True,
            )
        with pc2:
            st.markdown(
                metric_tile("\U0001f3e2", "UNTERNEHMENSGROESSE",
                            profile.get("stock_size_preference", "N/A")),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_tile("\U0001f30d", "GEO-FOKUS",
                            profile.get("geographic_focus", "N/A"), "purple"),
                unsafe_allow_html=True,
            )
        with pc3:
            st.markdown(
                metric_tile("\U0001f4ca", "MAX. POSITION",
                            f"{profile.get('max_position_size_pct', 20)}%"),
                unsafe_allow_html=True,
            )
            div_val = "Ja \U0001f4b0" if profile.get("dividend_preference") else "Nein"
            st.markdown(
                metric_tile("\U0001f4b0", "DIVIDENDEN", div_val),
                unsafe_allow_html=True,
            )

        sectors = profile.get("preferred_sectors", [])
        if sectors:
            st.markdown(
                tile("\U0001f3ed", "Bevorzugte Sektoren",
                     f'<div class="tile-value blue" style="font-size:16px;">{", ".join(sectors)}</div>'),
                unsafe_allow_html=True,
            )
        phil = profile.get("philosophy", "")
        if phil:
            st.markdown(
                tile("\U0001f4a1", "Anlagephilosophie",
                     f'<div style="color:#C0C8E0;font-style:italic;">"{phil}"</div>'),
                unsafe_allow_html=True,
            )
        adm = profile.get("admired_stocks", "")
        if adm:
            st.markdown(
                tile("\u2b50", "Bewunderte Investments",
                     f'<div style="color:#C0C8E0;">{adm}</div>'),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            tile("\U0001f4dd", "Profil einrichten",
                 '<div class="tile-sub">Fuell das Formular aus, damit die Agenten deinen Anlagestil kennen</div>',
                 glow="blue"),
            unsafe_allow_html=True,
        )

    st.markdown("### PROFIL BEARBEITEN")

    with st.form("profile_form"):
        f1, f2 = st.columns(2)
        with f1:
            risk = st.select_slider(
                "Risikobereitschaft", options=RISK_LEVELS,
                value=profile.get("risk_tolerance", "Mittel") if profile.get("risk_tolerance") in RISK_LEVELS else "Mittel",
                help="Wie viel Volatilitaet kannst du verkraften?",
            )
            horizon = st.selectbox(
                "Anlagehorizont", HORIZONS,
                index=HORIZONS.index(profile["investment_horizon"]) if profile.get("investment_horizon") in HORIZONS else 2,
            )
            stock_size = st.selectbox(
                "Unternehmensgroesse", STOCK_SIZES,
                index=STOCK_SIZES.index(profile["stock_size_preference"]) if profile.get("stock_size_preference") in STOCK_SIZES else 4,
            )
        with f2:
            geo = st.selectbox(
                "Geografischer Fokus", GEO_OPTIONS,
                index=GEO_OPTIONS.index(profile["geographic_focus"]) if profile.get("geographic_focus") in GEO_OPTIONS else 0,
            )
            max_pos = st.slider(
                "Max. Positionsgroesse (%)", min_value=5, max_value=50,
                value=profile.get("max_position_size_pct", 20),
            )
            div_pref = st.checkbox("Ich bevorzuge Dividendenaktien",
                                   value=profile.get("dividend_preference", False))

        sectors = st.multiselect(
            "Bevorzugte Sektoren", SECTOR_OPTIONS,
            default=[s for s in profile.get("preferred_sectors", []) if s in SECTOR_OPTIONS],
        )
        philosophy = st.text_area(
            "Anlagephilosophie", value=profile.get("philosophy", ""),
            placeholder="z.B. Ich investiere in disruptive Technologien mit 10x-Potenzial...",
            height=100,
        )
        admired = st.text_area(
            "Bewunderte Aktien/Unternehmen", value=profile.get("admired_stocks", ""),
            placeholder="z.B. SpaceX, Tesla, NVDA...",
            height=80,
        )
        submitted = st.form_submit_button("\U0001f4be Profil speichern", type="primary",
                                          use_container_width=True)
        if submitted:
            save_profile({
                "risk_tolerance": risk, "investment_horizon": horizon,
                "preferred_sectors": sectors, "stock_size_preference": stock_size,
                "geographic_focus": geo, "max_position_size_pct": max_pos,
                "dividend_preference": div_pref, "philosophy": philosophy,
                "admired_stocks": admired,
            })
            st.success("\u2705 Anlageprofil gespeichert!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    if st.button("\u26a0\ufe0f Profil zuruecksetzen"):
        reset_profile()
        st.success("\u2705 Profil zurueckgesetzt.")
        time.sleep(0.5)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# SEITE 6: EINSTELLUNGEN
# ═══════════════════════════════════════════════════════════════════

elif page == "\u2699\ufe0f Einstellungen":

    st.markdown(
        '<div class="page-header">'
        "<h1>\u2699\ufe0f Einstellungen</h1>"
        "<p>Konfiguriere deinen AI Portfolio Agent</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Settings as tiles
    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            tile("\U0001f511", "API Key",
                 '<div class="tile-sub">OpenAI API Zugang konfigurieren</div>'),
            unsafe_allow_html=True,
        )
        key_status = "\u2705 Konfiguriert" if check_api_key() else "\u274c Nicht konfiguriert"
        st.markdown(f"**Status:** {key_status}")
        new_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if st.button("\U0001f4be Key speichern", use_container_width=True):
            if new_key.strip():
                env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(f"OPENAI_API_KEY={new_key.strip()}\n")
                os.environ["OPENAI_API_KEY"] = new_key.strip()
                st.success("\u2705 API Key gespeichert!")
            else:
                st.warning("Bitte gueltigen Key eingeben.")

    with s2:
        st.markdown(
            tile("\U0001f916", "LLM-Modell",
                 '<div class="tile-sub">Welches Modell sollen die Agenten nutzen?</div>'),
            unsafe_allow_html=True,
        )
        selected = st.selectbox(
            "Modell", SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in SUPPORTED_MODELS else 0,
        )
        st.session_state.selected_model = selected
        st.info(f"Aktuelles Modell: **{selected}**")

    # Cache & Data tiles
    st.markdown("### CACHE & DATEN")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            tile("\U0001f9f9", "Analyse-Cache",
                 '<div class="tile-sub">Gespeicherte Analysen loeschen</div>'),
            unsafe_allow_html=True,
        )
        if st.button("Cache leeren", key="clear_cache", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.news_result = None
            st.session_state.portfolio_result = None
            st.success("\u2705 Cache geleert!")
    with d2:
        st.markdown(
            tile("\U0001f4ac", "Chat-Verlauf",
                 '<div class="tile-sub">Alle Chat-Nachrichten loeschen</div>'),
            unsafe_allow_html=True,
        )
        if st.button("Chat loeschen", key="clear_chat", use_container_width=True):
            clear_chat_history()
            st.session_state.chat_messages = []
            st.success("\u2705 Chat geloescht!")
    with d3:
        st.markdown(
            tile("\U0001f4bc", "Portfolio",
                 '<div class="tile-sub">Alle Positionen zuruecksetzen</div>'),
            unsafe_allow_html=True,
        )
        if st.button("Portfolio leeren", key="clear_portfolio", use_container_width=True):
            save_portfolio([])
            st.success("\u2705 Portfolio leer.")

    # About tile
    st.markdown("### UEBER DIESES PROJEKT")
    st.markdown(
        tile("\U0001f3af", "Skills & Technologien", """
        <table style="width:100%;margin-top:8px;">
            <tr><td style="padding:8px;"><b>\U0001f916 Multi-Agent</b></td><td style="padding:8px;">3 spezialisierte Agenten</td></tr>
            <tr><td style="padding:8px;"><b>\U0001f9e0 LLM</b></td><td style="padding:8px;">GPT-4o + Function Calling</td></tr>
            <tr><td style="padding:8px;"><b>\U0001f527 Tools</b></td><td style="padding:8px;">yfinance API Integration</td></tr>
            <tr><td style="padding:8px;"><b>\U0001f4ac Chat</b></td><td style="padding:8px;">Intent-Klassifikation + Routing</td></tr>
            <tr><td style="padding:8px;"><b>\U0001f9e0 Memory</b></td><td style="padding:8px;">Persistentes Nutzerprofil</td></tr>
            <tr><td style="padding:8px;"><b>\U0001f3a8 Design</b></td><td style="padding:8px;">Kachel-Dashboard + Glassmorphism</td></tr>
        </table>
        """),
        unsafe_allow_html=True,
    )
