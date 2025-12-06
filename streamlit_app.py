import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random
import re

# === NEW: external data helpers ===
import requests
from bs4 import BeautifulSoup
import pytz


# ========================= FINVIZ / FINTEL HELPERS =========================
def get_finviz_news_for_ticker(ticker: str, max_items: int = 12):
    """
    Scrape Finviz news headlines for a ticker.
    Handles bot-blocking by using a full browser fingerprint
    and retries if the first attempt fails.
    Returns list of dicts: {time, title, sent, url}
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://finviz.com/",
    }

    def fetch():
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                return None
            return BeautifulSoup(r.text, "html.parser")
        except Exception:
            return None

    soup = None
    for _ in range(3):  # up to 3 attempts
        soup = fetch()
        if soup:
            break

    if not soup:
        return []

    table = soup.select_one("table.fullview-news-outer")
    if not table:
        return []

    rows = table.select("tr")
    items = []

    for row in rows[:max_items]:
        parts = row.find_all("td")
        if len(parts) < 2:
            continue

        time_text = parts[0].get_text(strip=True)
        link_tag = parts[1].find("a")
        if not link_tag:
            continue

        title = link_tag.get_text(strip=True)
        news_url = "https://finviz.com/" + link_tag["href"].lstrip("/")

        lower = title.lower()
        if any(
            x in lower
            for x in [
                "up",
                "surge",
                "beat",
                "beats",
                "growth",
                "upgrade",
                "upgrades",
                "bull",
                "bullish",
                "record",
                "rally",
                "soar",
                "soars",
            ]
        ):
            sentiment = "🟢"
        elif any(
            x in lower
            for x in [
                "down",
                "fall",
                "falls",
                "miss",
                "misses",
                "plunge",
                "warning",
                "downgrade",
                "downgrades",
                "bear",
                "bearish",
                "cut",
                "cuts",
            ]
        ):
            sentiment = "🔴"
        else:
            sentiment = "⚪"

        items.append(
            {"time": time_text, "title": title, "sent": sentiment, "url": news_url}
        )

    return items


def get_finviz_news_today():
    """
    Pull general market headlines for today only, as on Finviz news page.
    """
    url = "https://finviz.com/news.ashx"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://finviz.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return []
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    today = datetime.now(pytz.timezone("US/Eastern")).strftime("%m-%d-%Y")
    items = []

    for row in soup.select("table.news-table tr"):
        date_td = row.select_one("td:nth-child(1)")
        link_td = row.select_one("td:nth-child(2) a")
        if not date_td or not link_td:
            continue

        date_text = date_td.get_text(strip=True)
        if today in date_text:
            items.append(
                {
                    "time": date_text,
                    "title": link_td.get_text(strip=True),
                    "url": "https://finviz.com/" + link_td["href"].lstrip("/"),
                }
            )
    return items


def get_fintel_short_data(ticker: str):
    """
    Scrape Fintel for a simple short-availability snapshot for the ticker.
    Returns dict {time, shares, fee} or None.
    """
    url = f"https://fintel.io/s/us/{ticker.lower()}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    short_info = {"time": None, "shares": None, "fee": None}

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        for row in rows[1:3]:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            time_txt = cells[0].get_text(strip=True)
            shares_txt = cells[1].get_text(strip=True)
            fee_txt = cells[2].get_text(strip=True)

            if not shares_txt:
                continue
            short_info["time"] = time_txt
            short_info["shares"] = shares_txt
            short_info["fee"] = fee_txt
            return short_info

    return None


# ========================= SETTINGS =========================
THREADS = 20  # keep high but not crazy
AUTO_REFRESH_DEFAULT = 120_000  # default auto-refresh every 120 seconds
HISTORY_LOOKBACK_DAYS = 10  # 🔥 10-day mode
INTRADAY_INTERVAL = "2m"  # 2-minute candles
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 5.0
DEFAULT_MIN_VOLUME = 100_000
DEFAULT_MIN_BREAKOUT = 0.0

# ========================= SESSION STATE FOR V11/V12 STREAMING =========================
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "auto_refresh_ms" not in st.session_state:
    st.session_state.auto_refresh_ms = AUTO_REFRESH_DEFAULT

# champion / seed universe state
if "seed_universe" not in st.session_state:
    # entries now look like: {"Symbol": "XYZ", "Exchange": "NASDAQ", "LastNewsSeed": "... UTC"}
    st.session_state.seed_universe = []
if "seed_universe_created_at" not in st.session_state:
    st.session_state.seed_universe_created_at = None
if "seed_universe_size" not in st.session_state:
    st.session_state.seed_universe_size = 0
if "seed_universe_mode" not in st.session_state:
    st.session_state.seed_universe_mode = None

# ========================= AUTO REFRESH (V11 streaming aware) =========================
if st.session_state.auto_refresh_enabled:
    st_autorefresh(interval=st.session_state.auto_refresh_ms, key="refresh_v11")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V12 – 10-Day Momentum Screener (Hybrid Volume/Randomized + ML/AI)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V12 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized + ML/AI)")
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Audio alerts • V9/V10/V11/V12 universe modes "
    "(classic / random / volume-ranked) • Live volume • ML-style probability • AI commentary"
)

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
        help="Example: AAPL, TSLA, NVDA, AMD",
    )

    max_universe = st.slider(
        "Max symbols to scan when no watchlist",
        min_value=2000,
        max_value=4000,
        value=2000,
        step=100,
        help="Keeps scans fast when you don't use a custom watchlist. Seed size is at least 2000.",
    )

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        options=[
            "Classic (Alphabetical Slice)",
            "Randomized Slice",
            "Live Volume Ranked (slower)",
        ],
        index=0,
        help=(
            "Classic = original behavior.\n"
            "Randomized = random subset of symbols each scan.\n"
            "Live Volume Ranked = prioritize highest intraday volume (slower)."
        ),
    )

    volume_rank_pool = st.slider(
        "Max symbols to consider when volume-ranking (V9)",
        min_value=100,
        max_value=2000,
        value=600,
        step=100,
        help="Used only when 'Live Volume Ranked (slower)' is selected.",
    )

    enable_enrichment = st.checkbox(
        "Include float/short + news (slower, more data)",
        value=False,
    )

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume = st.number_input("Min Daily Volume", 0, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 1.0)

    min_breakout_confirm = st.number_input(
        "Min Breakout Confirmation (0–100)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )

    min_entry_confidence = st.number_input(
        "Min Entry Confidence (0–100)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )

    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")

    catalyst_finviz_only = st.checkbox(
        "Finviz News Catalyst Required",
        value=False,
        help=(
            "PURE FINVIZ MODE: when enabled, all numeric filters are ignored and "
            "only tickers with current Finviz headlines are shown. "
            "Momentum stats still calculated for ranking."
        ),
    )

    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox(
        "Use Min Order Flow Bias Filter",
        value=False,
        help="When enabled, only keep symbols where buy volume dominates.",
    )
    min_ofb = st.slider(
        "Min Order Flow Bias (0–1, buyer control)",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="0.5 = equal buy/sell; 0.7 = strong buyer control.",
    )

    ignore_filters_for_watchlist = st.checkbox(
        "Ignore filters when watchlist is populated (V11)",
        value=False,
        help="When enabled and watchlist has symbols, hard filters (price, volume, etc.) are skipped.",
    )

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")

    enable_alerts = st.checkbox(
        "Enable Audio + Alert Banner",
        value=False,
        help="Turn this off to completely silence alerts.",
    )

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    st.markdown("---")
    st.subheader("V11/V12 Streaming Controls")
    auto_refresh_enabled = st.checkbox(
        "Enable Auto-Refresh (Streaming)",
        value=st.session_state.auto_refresh_enabled,
        help="Controls whether the app auto-refreshes. Takes effect on next refresh.",
    )
    auto_refresh_ms = st.slider(
        "Auto-Refresh Interval (ms)",
        min_value=10_000,
        max_value=300_000,
        value=st.session_state.auto_refresh_ms,
        step=5_000,
        help="Used when auto-refresh is enabled.",
    )
    st.session_state.auto_refresh_enabled = auto_refresh_enabled
    st.session_state.auto_refresh_ms = auto_refresh_ms

    preopen_mode = st.checkbox(
        "Pre-Open Scan Mode (V11)",
        value=False,
        help="Emphasize premarket moves & volume; de-emphasize longer-term trend.",
    )

    use_last_results = st.checkbox(
        "Use last scan results (no rescan, V11)",
        value=False,
        help="Use cached universe from prior run instead of rescanning.",
    )

    st.markdown("---")
    st.subheader("V12 Seeding")
    force_new_seed = st.button(
        "Force New Seed (V12)",
        help="Clear cached scans and reseed the universe on the next run.",
    )
    if force_new_seed:
        st.cache_data.clear()
        if "last_df" in st.session_state:
            del st.session_state["last_df"]
        st.session_state.seed_universe = []
        st.session_state.seed_universe_created_at = None
        st.session_state.seed_universe_size = 0
        st.session_state.seed_universe_mode = None
        if "alerted" in st.session_state:
            st.session_state.alerted = set()
        st.success("New seed will be used on the next scan.")

    if st.session_state.seed_universe_created_at:
        st.caption(
            f"Last seed: {st.session_state.seed_universe_created_at} • "
            f"Size: {st.session_state.seed_universe_size} • "
            f"Mode: {st.session_state.seed_universe_mode}"
        )

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        if "last_df" in st.session_state:
            del st.session_state["last_df"]
        st.success("Cache cleared — fresh scan will run now.")


# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    """
    Load US symbols (NASDAQ + otherlisted) in a robust way.
    Handles schema changes on nasdaqtrader with defensive column access.
    """
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|",
        engine="python",
        skipfooter=1,
        on_bad_lines="skip",
    )
    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|",
        engine="python",
        skipfooter=1,
        on_bad_lines="skip",
    )

    nasdaq_symbol_col = "Symbol"
    if nasdaq_symbol_col not in nasdaq.columns:
        nasdaq_symbol_col = nasdaq.columns[0]
    other_symbol_col = "ACT Symbol" if "ACT Symbol" in other.columns else other.columns[0]

    nasdaq_symbols = nasdaq[nasdaq_symbol_col].astype(str).str.strip()
    other_symbols = other[other_symbol_col].astype(str).str.strip()

    nasdaq_df = pd.DataFrame(
        {
            "Symbol": nasdaq_symbols,
            "Exchange": "NASDAQ",
        }
    )

    if "Exchange" in other.columns:
        other_exchange = other["Exchange"].fillna("NYSE/AMEX/ARCA").astype(str)
    else:
        other_exchange = pd.Series(["NYSE/AMEX/ARCA"] * len(other_symbols))

    other_df = pd.DataFrame(
        {
            "Symbol": other_symbols,
            "Exchange": other_exchange,
        }
    )

    df = pd.concat([nasdaq_df, other_df], ignore_index=True).dropna(subset=["Symbol"])
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")


def build_universe(
    watchlist_text: str,
    max_universe: int,
    universe_mode: str,
    volume_rank_pool: int,
):
    """
    Return a list of symbol dicts to scan, based on:
    - Watchlist
    - Cached champion seed universe (V12)
    - Classic alphabetical slice
    - Randomized slice
    - Live volume-ranked (V9)
    """
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n", " ").replace(",", " ").split()
        tickers = sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol": t, "Exchange": "WATCH"} for t in tickers]

    # ✅ V12: if we already have a seeded / champion universe, reuse it
    if st.session_state.seed_universe:
        # Only Symbol and Exchange are used; LastNewsSeed is ignored here
        return st.session_state.seed_universe[:max_universe]

    syms = load_symbols()

    # V9 modes (only used the first time, to create the seed_universe)
    if universe_mode == "Randomized Slice":
        base = syms[:]
        random.shuffle(base)
        universe = base[:max_universe]

    elif universe_mode == "Live Volume Ranked (slower)":
        base = syms[:volume_rank_pool]
        ranked = []
        for sym in base:
            try:
                t = yf.Ticker(sym["Symbol"])
                d = t.history(period="1d", interval=INTRADAY_INTERVAL, prepost=True)
                if not d.empty:
                    live_vol = float(d["Volume"].sum())
                    ranked.append({**sym, "LiveVol": live_vol})
            except Exception:
                continue

        if not ranked:
            universe = syms[:max_universe]
        else:
            ranked_sorted = sorted(
                ranked,
                key=lambda x: x.get("LiveVol", 0.0),
                reverse=True,
            )
            universe = ranked_sorted[:max_universe]
    else:
        # Classic
        universe = syms[:max_universe]

    # Store champion / seed universe in session so it persists across refreshes
    st.session_state.seed_universe = universe
    st.session_state.seed_universe_size = len(universe)
    st.session_state.seed_universe_mode = universe_mode
    st.session_state.seed_universe_created_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    return universe


# ========================= SCORING (10-DAY MODEL) =========================
def short_window_score(
    pm,
    yday,
    m3,
    m10,
    rsi7,
    rvol10,
    catalyst,
    squeeze,
    vwap,
    flow_bias,
    preopen_mode=False,
    vwap_enabled=True,
):
    """
    Short-window breakout score.
    """
    score = 0.0

    pm_w = 2.0 if preopen_mode else 1.6
    m10_w = 0.3 if preopen_mode else 0.6
    rvol_w = 2.6 if preopen_mode else 2.0

    if pm is not None:
        score += max(pm, 0) * pm_w
    if yday is not None:
        score += max(yday, 0) * 0.8
    if m3 is not None:
        score += max(m3, 0) * 1.2
    if m10 is not None:
        score += max(m10, 0) * m10_w

    if rsi7 is not None and rsi7 > 55:
        score += (rsi7 - 55) * 0.4

    if rvol10 is not None and rvol10 > 1.2:
        score += (rvol10 - 1.2) * rvol_w

    if vwap_enabled and vwap is not None and vwap > 0:
        score += min(vwap, 6) * 1.5

    if flow_bias is not None:
        score += (flow_bias - 0.5) * 22.0

    if catalyst:
        score += 8.0
    if squeeze:
        score += 12.0

    return round(score, 2)


def ml_breakout_probability(score: float, rvol10, pm, m10) -> float:
    """
    'ML-style' probability-like number, using a richer feature mix
    but kept lightweight (no external libraries).
    """
    try:
        base = score / 25.0
        if rvol10 is not None:
            base += (rvol10 - 1.0) * 0.15
        if pm is not None:
            base += (pm / 20.0) * 0.2
        if m10 is not None:
            base += (m10 / 50.0) * 0.1

        prob = 1 / (1 + math.exp(-base))
        return round(prob * 100, 1)
    except Exception:
        return None


def multi_timeframe_label(pm, m3, m10):
    """Simple multi-timeframe alignment label: intraday + 3D + 10D."""
    bull_intraday = pm is not None and pm > 0
    bull_3d = m3 is not None and m3 > 0
    bull_10d = m10 is not None and m10 > 0

    positives = sum([bull_intraday, bull_3d, bull_10d])

    if positives == 3:
        return "✅ Aligned Bullish (Intraday + 3D + 10D)"
    elif positives == 2:
        return "🟢 Leaning Bullish"
    elif positives == 1:
        return "🟡 Mixed"
    else:
        return "🔻 Not Aligned"


def news_sentiment_score(title: str, summary: str | None = None) -> float:
    """
    Very lightweight sentiment scorer using keywords.
    Returns value in roughly [-1, 1].
    """
    text = (title or "") + " " + (summary or "")
    text = text.lower()

    pos_words = [
        "beat",
        "beats",
        "strong",
        "surge",
        "upgrade",
        "upgrades",
        "bullish",
        "raises",
        "raise",
        "record",
        "jump",
        "rally",
        "soars",
        "soar",
        "momentum",
    ]
    neg_words = [
        "miss",
        "misses",
        "weak",
        "downgrade",
        "downgrades",
        "bearish",
        "cuts",
        "cut",
        "plunge",
        "fall",
        "falls",
        "tumbles",
        "tumble",
        "guidance cut",
        "warning",
    ]

    score = 0
    for w in pos_words:
        if w in text:
            score += 1
    for w in neg_words:
        if w in text:
            score -= 1

    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 5.0))


def entry_confidence_score(vwap_dist, rvol10, flow_bias) -> float:
    """
    Entry timing confidence 0–100 based on VWAP distance, RVOL, and OFB.
    """
    if vwap_dist is None or rvol10 is None or flow_bias is None:
        return 50.0  # neutral

    score = 60.0

    if -1 <= vwap_dist <= 3:
        score += 15
    elif abs(vwap_dist) > 8:
        score -= 15

    if rvol10 > 2:
        score += 10
    elif rvol10 < 0.7:
        score -= 10

    score += (flow_bias - 0.5) * 40.0

    return round(max(0.0, min(100.0, score)), 1)


def breakout_confirmation_index(score, rvol10, pm, m10) -> float:
    """
    Breakout confirmation index 0–100 combining score, RVOL, PM, and 10D trend.
    """
    base = (score or 0) / 2.0
    if rvol10 is not None:
        base += max(0.0, (rvol10 - 1.0) * 8.0)
    if pm is not None:
        base += max(0.0, pm) * 1.2
    if m10 is not None and m10 > 0:
        base += min(m10, 30) * 0.8

    return round(max(0.0, min(100.0, base)), 1)


# ========================= SIMPLE AI COMMENTARY =========================
def ai_commentary(score, pm, rvol, flow_bias, vwap, ten_day, sentiment, entry_conf, bci, preopen_mode):
    comments = []

    if score is not None:
        if score >= 90:
            comments.append("Explosive momentum profile, risk-on candidate.")
        elif score >= 60:
            comments.append("Constructive momentum with improving structure.")
        elif score >= 30:
            comments.append("Early momentum, still needs confirmation.")

    if pm is not None:
        if pm > 5:
            comments.append("Strong premarket demand showing early accumulation.")
        elif pm < -3:
            comments.append("Notable premarket supply; caution on chasing intraday pops.")

    if rvol is not None:
        if rvol > 2:
            comments.append("Volume aggressively expanding vs 10-day baseline.")
        elif rvol < 0.7:
            comments.append("Liquidity muted; slippage/whipsaws more likely.")

    if flow_bias is not None:
        if flow_bias > 0.7:
            comments.append("Buyers dominating tape, dips may get absorbed quickly.")
        elif flow_bias < 0.4:
            comments.append("Sellers pressing, rallies could be sold into.")

    if vwap is not None:
        if 0 <= vwap <= 3:
            comments.append("Trading near/just above VWAP – healthy risk/reward zone.")
        elif vwap > 5:
            comments.append("Extended well above VWAP – momentum but risk of chase.")
        elif vwap < 0:
            comments.append("Below VWAP – still under distribution pressure.")

    if ten_day is not None:
        if ten_day > 15:
            comments.append("10D structure confirmed uptrend; pullbacks may be buyable.")
        elif ten_day < -8:
            comments.append("10D trend in clear distribution – countertrend risk.")

    if sentiment is not None:
        if sentiment > 0.4:
            comments.append("Headline flow skewed positive; narrative supportive.")
        elif sentiment < -0.4:
            comments.append("Recent headlines skewed negative; narrative drag present.")

    if preopen_mode:
        comments.append("Pre-open mode: signal weights biased toward PM and early volume.")

    comments.append(f"Entry confidence ~ {entry_conf:.0f}/100.")
    comments.append(f"Breakout confirmation ~ {bci:.0f}/100.")

    if not comments:
        return "Neutral / indecisive tape — watching for clearer confirmation."

    return " | ".join(comments)


# ========================= CORE SCAN =========================
def scan_one(
    sym,
    enable_enrichment: bool,
    enable_ofb_filter: bool,
    min_ofb: float,
    preopen_mode: bool,
    vwap_enabled_flag: bool,
):
    try:
        ticker = sym["Symbol"]
        exchange = sym.get("Exchange", "UNKNOWN")
        stock = yf.Ticker(ticker)

        # Daily 10d history
        hist = stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d", interval="1d")
        if hist is None or hist.empty or len(hist) < 5:
            return None

        close = hist["Close"]
        daily_volume = hist["Volume"]

        price = float(close.iloc[-1])
        daily_vol_last = float(daily_volume.iloc[-1])

        # Daily volume filter for stability
        if price > max_price or daily_vol_last < min_volume:
            return None

        premarket_pct = None
        vwap_dist = None
        order_flow_bias = None
        live_intraday_volume = None

        try:
            intra = stock.history(period=INTRADAY_RANGE, interval=INTRADAY_INTERVAL, prepost=True)
        except Exception:
            intra = None

        if intra is not None and not intra.empty and len(intra) >= 3:
            iclose = intra["Close"]
            iopen = intra["Open"]
            ivol = intra["Volume"]

            live_intraday_volume = float(ivol.sum())

            last_close = float(iclose.iloc[-1])
            prev_close_intraday = float(iclose.iloc[-2])
            if prev_close_intraday > 0:
                premarket_pct = (last_close - prev_close_intraday) / prev_close_intraday * 100

            typical_price = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            total_vol = ivol.sum()
            if total_vol > 0:
                vwap_val = float((typical_price * ivol).sum() / total_vol)
                if vwap_val > 0:
                    vwap_dist = (price - vwap_val) / vwap_val * 100

            of_df = intra[["Open", "Close", "Volume"]].dropna()
            if not of_df.empty:
                sign = (of_df["Close"] > of_df["Open"]).astype(int) - (of_df["Close"] < of_df["Open"]).astype(int)
                buy_vol = float((of_df["Volume"] * (sign > 0)).sum())
                sell_vol = float((of_df["Volume"] * (sign < 0)).sum())
                total = buy_vol + sell_vol
                if total > 0:
                    order_flow_bias = buy_vol / total

        try:
            fi = stock.fast_info
            pre_price = fi.get("last_price", None)
            prev_close = fi.get("regular_market_previous_close", None)

            if pre_price and prev_close and prev_close > 0:
                calc_pm = (pre_price - prev_close) / prev_close * 100

                now = datetime.now(timezone.utc)
                if now.hour < 14 or (now.hour == 14 and now.minute < 30):
                    premarket_pct = round(calc_pm, 2)
        except Exception:
            pass

        if live_intraday_volume is None:
            live_intraday_volume = daily_vol_last

        if len(close) >= 2 and close.iloc[-2] > 0:
            yday_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        else:
            yday_pct = None

        if len(close) >= 4 and close.iloc[-4] > 0:
            m3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100
        else:
            m3 = None

        if close.iloc[0] > 0:
            m10 = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        else:
            m10 = None

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi7 = float(rsi_series.iloc[-1])

        ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
        ema_trend = "🔥 Breakout" if price > ema10 and rsi7 > 55 else "Neutral"

        avg10 = float(daily_volume.mean()) if len(daily_volume) > 0 else 0
        rvol10 = live_intraday_volume / avg10 if avg10 > 0 else None

        if enable_ofb_filter:
            if order_flow_bias is None or order_flow_bias < min_ofb:
                return None

        squeeze = False
        low_float = False
        catalyst = False
        sector = "Unknown"
        industry = "Unknown"
        short_pct_display = None
        sentiment_score_val = 0.0

        if enable_enrichment:
            try:
                info = stock.get_info() or {}
                float_shares = info.get("floatShares")
                short_pct = info.get("shortPercentOfFloat")
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")

                low_float = bool(float_shares and float_shares < 20_000_000)
                squeeze = bool(short_pct and short_pct > 0.15)
                short_pct_display = round(short_pct * 100, 2) if short_pct else None
            except Exception:
                pass

            try:
                news = stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub = datetime.fromtimestamp(news[0]["providerPublishTime"], tz=timezone.utc)
                    catalyst = (datetime.now(timezone.utc) - pub).days <= 3

                sent_vals = []
                for n in news[:5]:
                    t = n.get("title", "")
                    s = n.get("summary", "")
                    sent_vals.append(news_sentiment_score(t, s))

                if sent_vals:
                    sentiment_score_val = round(sum(sent_vals) / len(sent_vals), 2)
                else:
                    sentiment_score_val = 0.0

            except Exception:
                pass

        mtf_label = multi_timeframe_label(premarket_pct, m3, m10)

        score = short_window_score(
            pm=premarket_pct,
            yday=yday_pct,
            m3=m3,
            m10=m10,
            rsi7=rsi7,
            rvol10=rvol10,
            catalyst=catalyst,
            squeeze=squeeze,
            vwap=vwap_dist,
            flow_bias=order_flow_bias,
            preopen_mode=preopen_mode,
            vwap_enabled=vwap_enabled_flag,
        )
        prob_rise = ml_breakout_probability(score, rvol10, premarket_pct, m10)

        entry_conf = entry_confidence_score(vwap_dist, rvol10, order_flow_bias)
        bci = breakout_confirmation_index(score, rvol10, premarket_pct, m10)

        ai_text = ai_commentary(
            score=score,
            pm=premarket_pct,
            rvol=rvol10,
            flow_bias=order_flow_bias,
            vwap=vwap_dist,
            ten_day=m10,
            sentiment=sentiment_score_val,
            entry_conf=entry_conf,
            bci=bci,
            preopen_mode=preopen_mode,
        )

        spark_series = close

        return {
            "Symbol": ticker,
            "Exchange": exchange,
            "Price": round(price, 2),
            "Volume": int(live_intraday_volume),
            "Score": score,
            "Prob_Rise%": prob_rise,
            "PM%": round(premarket_pct, 2) if premarket_pct is not None else None,
            "YDay%": round(yday_pct, 2) if yday_pct is not None else None,
            "3D%": round(m3, 2) if m3 is not None else None,
            "10D%": round(m10, 2) if m10 is not None else None,
            "RSI7": round(rsi7, 2),
            "EMA10 Trend": ema_trend,
            "RVOL_10D": round(rvol10, 2) if rvol10 is not None else None,
            "VWAP%": round(vwap_dist, 2) if vwap_dist is not None else None,
            "FlowBias": round(order_flow_bias, 2) if order_flow_bias is not None else None,
            "Squeeze?": squeeze,
            "LowFloat?": low_float,
            "Short % Float": short_pct_display,
            "Sector": sector,
            "Industry": industry,
            "Catalyst": catalyst,
            "MTF_Trend": mtf_label,
            "Spark": spark_series,
            "AI_Commentary": ai_text,
            "Sentiment": round(sentiment_score_val, 2),
            "Entry_Confidence": entry_conf,
            "Breakout_Confirm": bci,
        }

    except Exception:
        return None


@st.cache_data(ttl=6)
def run_scan(
    watchlist_text: str,
    max_universe: int,
    enable_enrichment: bool,
    enable_ofb_filter: bool,
    min_ofb: float,
    universe_mode: str,
    volume_rank_pool: int,
    preopen_mode: bool,
    ignore_filters_for_watchlist_flag: bool,
    vwap_enabled_flag: bool,
):
    """
    V11/V12 lightning engine.
    """
    universe = build_universe(
        watchlist_text,
        max_universe,
        universe_mode,
        volume_rank_pool,
    )
    results = []

    global min_volume, max_price
    saved_min_volume = min_volume
    saved_max_price = max_price

    if ignore_filters_for_watchlist_flag and watchlist_text.strip():
        min_volume = 0
        max_price = 10_000.0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
            futures = [
                ex.submit(
                    scan_one,
                    sym,
                    enable_enrichment,
                    enable_ofb_filter,
                    min_ofb,
                    preopen_mode,
                    vwap_enabled_flag,
                )
                for sym in universe
            ]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res:
                    results.append(res)
    finally:
        min_volume = saved_min_volume
        max_price = saved_max_price

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ========================= SPARKLINE & CHART HELPERS =========================
def sparkline(series: pd.Series):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=series.values,
            mode="lines",
            line=dict(width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=60,
        width=160,
        margin=dict(l=2, r=2, t=2, b=2),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def bigline(series: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=series.values,
            mode="lines+markers",
            name=title,
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Bars (last 10 days)",
        yaxis_title="Price",
    )
    return fig


# ========================= AUDIO ALERT STATE =========================
if "alerted" not in st.session_state:
    st.session_state.alerted = set()


def trigger_audio_alert(symbol: str, reason: str):
    """Play sound + mark symbol as alerted once per session."""
    st.session_state.alerted.add(symbol)
    audio_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    st.warning(f"🔔 {symbol}: {reason}")


# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning (10-day momentum, V12 hybrid universe)…"):
    if use_last_results and "last_df" in st.session_state:
        df_raw = st.session_state["last_df"].copy()
    else:
        effective_universe_mode = universe_mode
        if catalyst_finviz_only and not watchlist_text.strip():
            effective_universe_mode = "Live Volume Ranked (slower)"

        df_raw = run_scan(
            watchlist_text,
            max_universe,
            enable_enrichment,
            enable_ofb_filter,
            min_ofb,
            effective_universe_mode,
            volume_rank_pool,
            preopen_mode,
            ignore_filters_for_watchlist,
            vwap_only,
        )
        st.session_state["last_df"] = df_raw.copy()

if df_raw.empty:
    st.error("No results found. Try adding a watchlist or relaxing filters.")
else:
    finviz_cache: dict[str, list[dict]] = {}
    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    seed_map = {entry["Symbol"]: entry for entry in st.session_state.seed_universe}

    for sym in df_raw["Symbol"].unique():
        try:
            items = get_finviz_news_for_ticker(sym)
        except Exception:
            items = []
        finviz_cache[sym] = items

        if items:
            if sym in seed_map:
                seed_map[sym]["LastNewsSeed"] = now_utc_str
            else:
                try:
                    ex = df_raw.loc[df_raw["Symbol"] == sym, "Exchange"].iloc[0]
                except Exception:
                    ex = "UNKNOWN"
                entry = {
                    "Symbol": sym,
                    "Exchange": ex,
                    "LastNewsSeed": now_utc_str,
                }
                st.session_state.seed_universe.append(entry)
                seed_map[sym] = entry

    st.session_state.seed_universe_size = len(st.session_state.seed_universe)

    df_raw["FinvizNews"] = df_raw["Symbol"].map(lambda s: bool(finviz_cache.get(s)))
    df_raw["LastNewsSeed"] = df_raw["Symbol"].map(
        lambda s: seed_map.get(s, {}).get("LastNewsSeed")
    )

    if catalyst_finviz_only:
        df = df_raw[df_raw["FinvizNews"]].copy()
    else:
        df = df_raw.copy()

        if not (ignore_filters_for_watchlist and watchlist_text.strip()):
            df = df[df["Score"] >= min_breakout]

            if min_pm_move != 0.0:
                df = df[df["PM%"].fillna(-999) >= min_pm_move]
            if min_yday_gain != 0.0:
                df = df[df["YDay%"].fillna(-999) >= min_yday_gain]
            if squeeze_only:
                df = df[df["Squeeze?"]]
            if catalyst_only:
                df = df[df["Catalyst"]]
            if vwap_only:
                df = df[df["VWAP%"].fillna(-999) > 0]

            if min_breakout_confirm > 0.0 and "Breakout_Confirm" in df.columns:
                df = df[df["Breakout_Confirm"].fillna(-1) >= min_breakout_confirm]

            if min_entry_confidence > 0.0 and "Entry_Confidence" in df.columns:
                df = df[df["Entry_Confidence"].fillna(-1) >= min_entry_confidence]

    if df.empty:
        st.error("No results left after filters. Try relaxing constraints.")
    else:
        df = df.sort_values(
            by=["Score", "FinvizNews", "PM%", "RSI7"],
            ascending=[False, False, False, False],
        )

        st.subheader(f"🔥 10-Day Momentum Board (V12) — {len(df)} symbols")

        if enable_alerts and st.session_state.alerted:
            alerted_list = ", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        for _, row in df.iterrows():
            sym = row["Symbol"]
            finviz_items = finviz_cache.get(sym, [])
            has_finviz = bool(finviz_items)
            last_news_seed = row.get("LastNewsSeed", None)

            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym, f"Score {row['Score']}")
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym, f"Premarket {row['PM%']}%")
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym, f"VWAP Dist {row['VWAP%']}%")

            c1, c2, c3, c4 = st.columns([2, 3, 3, 3])

            c1.markdown(f"**{sym}** ({row['Exchange']})")

            if has_finviz:
                c1.markdown("🔥 **Catalyst (Finviz)**")
            else:
                c1.markdown("⚪ No Finviz Catalyst")

            c1.write(f"💲 Price: {row['Price']}")
            c1.write(f"📊 Live Volume: {row['Volume']:,}")
            c1.write(f"🔥 Score: **{row['Score']}**")
            c1.write(f"🤖 ML Prob_Rise: {row['Prob_Rise%']}%")
            c1.write(f"{row['MTF_Trend']}")
            c1.write(f"Trend: {row['EMA10 Trend']}")

            if last_news_seed:
                c1.caption(f"Last News Seeded: {last_news_seed}")
            else:
                c1.caption("Last News Seeded: —")

            price_val = float(row["Price"])
            bci_val = row.get("Breakout_Confirm", 0.0)
            entry_val = row.get("Entry_Confidence", 0.0)

            try:
                if bci_val is None or pd.isna(bci_val):
                    bci_val = 0.0
                if entry_val is None or pd.isna(entry_val):
                    entry_val = 0.0
            except Exception:
                bci_val = bci_val or 0.0
                entry_val = entry_val or 0.0

            ai_target = round(price_val * (1 + (bci_val / 250.0) + (entry_val / 400.0)), 2)
            ai_stop = round(price_val * (1 - (1 - entry_val / 100.0) * 0.05), 2)

            c1.write(f"🎯 AI Target: **${ai_target}**")
            c1.write(f"🛑 AI Stop: **${ai_stop}**")

            try:
                rr = (ai_target - price_val) / max(0.01, (price_val - ai_stop))
                rr_text = f"{rr:.2f} : 1"
            except Exception:
                rr = None
                rr_text = "—"

            c1.write(f"📈 R:R: **{rr_text}**")

            ai_expl_list = []

            if bci_val >= 70:
                ai_expl_list.append("Breakout structure strongly confirmed.")
            elif bci_val >= 50:
                ai_expl_list.append("Moderate breakout confirmation present.")
            else:
                ai_expl_list.append("Weak confirmation — target conservative.")

            if entry_val >= 70:
                ai_expl_list.append("Entry confidence high; tape favoring long entries.")
            elif entry_val >= 50:
                ai_expl_list.append("Entry timing acceptable.")
            else:
                ai_expl_list.append("Entry window uncertain; volatility elevated.")

            if row["VWAP%"] is not None:
                if row["VWAP%"] > 0:
                    ai_expl_list.append("Price holding above VWAP (bullish positioning).")
                else:
                    ai_expl_list.append("Below VWAP — higher risk of failed breakout.")

            if row["10D%"] is not None:
                if row["10D%"] > 0:
                    ai_expl_list.append("10-day trend supportive.")
                else:
                    ai_expl_list.append("10-day trend weak — target reduced.")

            flow = row.get("FlowBias", None)
            if flow is not None:
                if flow > 0.6:
                    ai_expl_list.append("Buyers absorbing dips; strong participation.")
                elif flow < 0.4:
                    ai_expl_list.append("Sellers active — cautious stop placement.")

            ai_target_expl = " ".join(ai_expl_list)
            c1.markdown(f"🧠 **AI Target Rationale:** {ai_target_expl}")

            c2.write(f"PM%: {row['PM%']}")
            c2.write(f"YDay%: {row['YDay%']}")
            c2.write(f"3D%: {row['3D%']}  |  10D%: {row['10D%']}")
            c2.write(f"RSI7: {row['RSI7']}  |  RVOL_10D: {row['RVOL_10D']}x")
            c2.write(f"Breakout Confirm: {row.get('Breakout_Confirm', 0)} / 100")
            c2.write(f"Entry Confidence: {row.get('Entry_Confidence', 0)} / 100")

            c3.write(f"VWAP Dist %: {row['VWAP%']}")
            c3.write(f"Order Flow Bias: {row['FlowBias']}")
            if enable_enrichment:
                c3.write(
                    f"Squeeze: {row['Squeeze?']} | LowFloat: {row['LowFloat?']}"
                )
                c3.write(f"Sec/Ind: {row['Sector']} / {row['Industry']}")
                c3.write(f"News Sentiment: {row.get('Sentiment', 0)}")
            else:
                c3.write("Enrichment: OFF (float/short/news skipped for speed)")

            try:
                short_data = get_fintel_short_data(sym)
            except Exception:
                short_data = None

            if short_data and (short_data.get("shares") or short_data.get("fee")):
                shares_txt = short_data.get("shares") or "—"
                fee_txt = short_data.get("fee") or "—"
                time_txt = short_data.get("time")
                c3.write(f"Shortable (Fintel): {shares_txt}")
                c3.write(f"Borrow Fee (Fintel): {fee_txt}")
                if time_txt:
                    c3.write(f"Short Data Time: {time_txt}")
            else:
                c3.write("Shortable (Fintel): n/a")

            c3.markdown(f"🧠 **AI View:** {row.get('AI_Commentary', '—')}")

            with c3.expander("📰 Recent Headlines (Finviz)", expanded=True):
                if not finviz_items:
                    st.write("No recent Finviz headlines found for this ticker.")
                else:
                    for n in finviz_items:
                        st.markdown(
                            f"{n['sent']} "
                            f"[{n['title']}]({n['url']})  \n"
                            f"<span style='font-size:10px;color:gray'>{n['time']}</span>",
                            unsafe_allow_html=True,
                        )

            c4.plotly_chart(sparkline(row["Spark"]), use_container_width=False)
            with c4.expander("📊 View 10-day chart"):
                c4.plotly_chart(bigline(row["Spark"], f"{sym} - Last 10 Days"), use_container_width=True)

            st.divider()

        raw_watch = watchlist_text.strip()
        if raw_watch:
            raw = raw_watch.replace("\n", " ").replace(",", " ").split()
            wl_tickers = sorted(set(s.upper() for s in raw if s.strip()))
            wl_df = df[df["Symbol"].isin(wl_tickers)]

            if not wl_df.empty:
                st.subheader("📋 Watchlist Multi-View (V11/V12)")
                st.dataframe(
                    wl_df[
                        [
                            "Symbol",
                            "Price",
                            "Volume",
                            "Score",
                            "Prob_Rise%",
                            "PM%",
                            "10D%",
                            "RVOL_10D",
                            "VWAP%",
                            "FlowBias",
                            "Breakout_Confirm",
                            "Entry_Confidence",
                        ]
                    ],
                    use_container_width=True,
                )

        csv_cols = [
            "Symbol",
            "Exchange",
            "Price",
            "Volume",
            "Score",
            "Prob_Rise%",
            "PM%",
            "YDay%",
            "3D%",
            "10D%",
            "RSI7",
            "EMA10 Trend",
            "RVOL_10D",
            "VWAP%",
            "FlowBias",
            "Squeeze?",
            "LowFloat?",
            "Short % Float",
            "Sector",
            "Industry",
            "Catalyst",
            "MTF_Trend",
            "AI_Commentary",
            "Sentiment",
            "Entry_Confidence",
            "Breakout_Confirm",
            "FinvizNews",
            "LastNewsSeed",
        ]
        csv_cols = [c for c in csv_cols if c in df.columns]

        st.download_button(
            "📥 Download Screener CSV",
            data=df[csv_cols].to_csv(index=False),
            file_name="v12_10day_momentum_screener_hybrid_ml_ai.csv",
            mime="text/csv",
        )

        with st.expander("📰 Today's Market Headlines (Finviz)"):
            try:
                finviz_news_today = get_finviz_news_today()
                if not finviz_news_today:
                    st.write("No Finviz headlines found for today yet.")
                else:
                    for n in finviz_news_today:
                        st.markdown(f"**{n['time']}** — [{n['title']}]({n['url']})")
            except Exception:
                st.write("⚠ Could not fetch Finviz daily headlines.")

st.caption("For research and education only. Not financial advice.")




