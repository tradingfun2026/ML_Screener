import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random
import re
import json

# === NEW: external data helpers ===
import requests
from bs4 import BeautifulSoup
import pytz

# ========================= FINVIZ / FINTEL HELPERS =========================
def get_finviz_news_for_ticker(ticker: str, max_items: int = 12):
    """
    Scrape ONLY today's Finviz news headlines for a ticker.
    - Filters strictly to current US/Eastern calendar date
    - Adds a 'breaking' flag for headlines < 20 minutes old

    Returns list of dicts: {time, title, sent, url, breaking}
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
    for _ in range(3):
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

    # Current date/time in Finviz timezone
    et_tz = pytz.timezone("US/Eastern")
    et_now = datetime.now(et_tz)
    today_date = et_now.date()

    # Finviz only stamps the FIRST row of each date block with a date; later
    # rows show only the time and inherit the most recent date above them.
    current_date = None

    for row in rows[:max_items]:
        tds = row.find_all("td")
        if len(tds) < 2:
            continue

        time_text = tds[0].get_text(strip=True)
        headline_tag = tds[1].find("a")
        if not headline_tag:
            continue

        parts = time_text.split()
        if len(parts) >= 2 and "-" in parts[0]:
            try:
                current_date = datetime.strptime(parts[0], "%b-%d-%y").date()
            except Exception:
                continue
        else:
            # Time-only row — drop if we haven't yet seen a dated row above it
            if current_date is None:
                continue

        # Filter by calendar date (today only)
        if current_date != today_date:
            continue

        title = headline_tag.get_text(strip=True)
        news_url = "https://finviz.com/" + headline_tag["href"].lstrip("/")

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
            {
                "time": time_text,
                "title": title,
                "sent": sentiment,
                "url": news_url,
            }
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

# A Finviz headline is flagged as NEW/breaking only if this session first
# observed its URL within this many seconds. Anchors "new" to our scan time
# rather than Finviz's publish timestamp, so stale Friday news doesn't fire
# a fresh badge on Monday and we don't badge a headline whose price has
# already moved.
BREAKING_WINDOW_SECONDS = 180

# NEW: TSX symbol list (Canada)
TSX_INSTRUMENTS_URL = (
    "https://github.com/LondonMarket/Global-Stock-Symbols/raw/main/"
    "tse_instrument_list_june_2024.xlsx"
)

# ========================= SESSION STATE FOR V11/V12 STREAMING =========================
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "auto_refresh_ms" not in st.session_state:
    st.session_state.auto_refresh_ms = AUTO_REFRESH_DEFAULT

# champion / seed universe state
if "seed_universe" not in st.session_state:
    # entries can have: {"Symbol": "XYZ", "Exchange": "NASDAQ", "LastNewsSeed": "... UTC"}
    st.session_state.seed_universe = []
if "seed_universe_created_at" not in st.session_state:
    st.session_state.seed_universe_created_at = None
if "seed_universe_size" not in st.session_state:
    st.session_state.seed_universe_size = 0
if "seed_universe_mode" not in st.session_state:
    st.session_state.seed_universe_mode = None

# Tracks {url: datetime first observed by this session}.
# Used to decide which headlines deserve the NEW badge.
if "first_seen_finviz_urls" not in st.session_state:
    st.session_state.first_seen_finviz_urls = {}
# The first Finviz pass of a session has no history to compare against, so
# every headline would otherwise appear "new". We seed the dict on that
# pass and only badge new arrivals on subsequent passes.
if "finviz_warmup_done" not in st.session_state:
    st.session_state.finviz_warmup_done = False

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
            "PURE FINVIZ MODE: when enabled, filters to US tickers with today-only "
            "Finviz headlines. Canadian tickers are still shown but marked as "
            "'Finviz not available for CAD ticker'."
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
    Load North American symbols:
    - US symbols (NASDAQ + otherlisted) in a robust way.
    - Canadian TSX symbols from a public instrument list.
    Handles schema changes on nasdaqtrader with defensive column access.
    """
    # ---------- US UNIVERSE ----------
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
    # Keep plain US-style tickers
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]

    # ---------- CANADIAN UNIVERSE (TSX) ----------
    try:
        tse = pd.read_excel(TSX_INSTRUMENTS_URL)
        cand_symbol_col = None
        for col in ["Symbol", "Ticker", "TSX", "TSX_TICKER"]:
            if col in tse.columns:
                cand_symbol_col = col
                break
        if cand_symbol_col is None:
            cand_symbol_col = tse.columns[0]

        tse_symbols = tse[cand_symbol_col].astype(str).str.strip()
        tse_df = pd.DataFrame(
            {
                "Symbol": tse_symbols,
                "Exchange": "TSX",
            }
        )
        tse_df = tse_df[tse_df["Symbol"].str.len() > 0]
        df = pd.concat([df, tse_df], ignore_index=True)
    except Exception:
        # If TSX list fails, fall back to US-only
        pass

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
        return st.session_state.seed_universe[:max_universe]

    syms = load_symbols()

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
def _capped_signed(value, cap_pos, cap_neg, pos_w, neg_w):
    """Apply asymmetric caps + weights to a signed momentum value.

    Bounds the contribution so a single outlier (e.g. a 500% premarket
    runner) can't dominate the score, and preserves negative information
    so a crashing stock no longer scores the same as a flat one.
    """
    if value is None:
        return 0.0
    if value >= 0:
        return min(value, cap_pos) * pos_w
    return max(value, -cap_neg) * neg_w


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
    Short-window breakout score. All momentum terms are capped so the
    output stays on a comparable scale across days and across the
    penny-stock outliers this scanner targets.
    """
    score = 0.0

    pm_w = 2.0 if preopen_mode else 1.6
    m10_w = 0.3 if preopen_mode else 0.6
    rvol_w = 2.6 if preopen_mode else 2.0

    # Momentum terms: positives capped, negatives penalized at half-weight.
    score += _capped_signed(pm,   cap_pos=30, cap_neg=15, pos_w=pm_w,  neg_w=pm_w * 0.5)
    score += _capped_signed(yday, cap_pos=20, cap_neg=10, pos_w=0.8,   neg_w=0.4)
    score += _capped_signed(m3,   cap_pos=30, cap_neg=15, pos_w=1.2,   neg_w=0.6)
    score += _capped_signed(m10,  cap_pos=50, cap_neg=25, pos_w=m10_w, neg_w=m10_w * 0.5)

    # RSI(7) bonus capped at RSI 72 — past that it's overbought, not stronger.
    if rsi7 is not None and rsi7 > 55:
        score += min(rsi7 - 55, 17) * 0.4

    # RVOL bonus capped at 5x avg (anything beyond is already extreme).
    if rvol10 is not None and rvol10 > 1.2:
        score += min(rvol10 - 1.2, 3.8) * rvol_w

    if vwap_enabled and vwap is not None and vwap > 0:
        score += min(vwap, 6) * 1.5

    if flow_bias is not None:
        score += (flow_bias - 0.5) * 22.0

    if catalyst:
        score += 8.0
    if squeeze:
        score += 12.0

    return round(score, 2)


def momentum_index(score: float, rvol10, pm, m10) -> float:
    """
    0-100 momentum index. NOT a probability — just a smoothed, bounded
    transform of the breakout score and key confirming factors. Higher
    means stronger momentum profile.

    Renamed from ml_breakout_probability: the old name implied a
    statistical likelihood that was never trained or calibrated. The
    denominator is now sized to the post-cap score range so the index
    actually spreads across 0-100 instead of saturating above ~score 50.
    """
    try:
        base = score / 80.0
        if rvol10 is not None:
            base += (rvol10 - 1.0) * 0.15
        if pm is not None:
            base += (pm / 20.0) * 0.2
        if m10 is not None:
            base += (m10 / 50.0) * 0.1

        idx = 1 / (1 + math.exp(-base))
        return round(idx * 100, 1)
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
    (Kept for compatibility; no longer used for Finviz sentiment.)
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


def finviz_sentiment_from_items(items: list[dict]) -> float:
    """
    Compute sentiment from Finviz headline items ONLY, using the
    same keyword idea as news_sentiment_score. Returns ~[-1, 1].
    """
    if not items:
        return 0.0

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
        "growth",
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
        "probe",
        "investigation",
    ]

    score = 0
    for n in items[:10]:
        txt = (n.get("title", "") or "").lower()
        for w in pos_words:
            if w in txt:
                score += 1
        for w in neg_words:
            if w in txt:
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


def breakout_confirmation_index(
    last_price,
    hist_close,
    hist_volume,
    intra_volume_today,
    vwap_dist,
) -> float:
    """
    0-100 breakout confirmation built from features that are NOT already
    inside `short_window_score`, so it's a genuine second opinion rather
    than a rescaled echo:

      - Proximity to / breach of the prior 10-day high (price structure)
      - Today's intraday volume vs YESTERDAY's full-day volume (a raw
        burst signal, not the smoothed 10-day RVOL the score already uses)
      - Currently holding above intraday VWAP

    Returns None when there's too little data to evaluate.
    """
    if hist_close is None or len(hist_close) < 2 or last_price is None:
        return None

    score = 50.0  # neutral baseline

    # 1. Proximity to / breach of the prior 10-day high (excluding today)
    try:
        max_prior = float(hist_close.iloc[:-1].max())
    except Exception:
        max_prior = None
    if max_prior and max_prior > 0:
        breakout_pct = (last_price - max_prior) / max_prior * 100
        if breakout_pct > 2:
            score += 25       # clear breakout above the range
        elif breakout_pct > 0:
            score += 15       # marginal breakout
        elif breakout_pct > -2:
            score += 8        # coiling just under the high
        elif breakout_pct > -5:
            score -= 5
        else:
            score -= 15       # well below the recent ceiling

    # 2. Today's intraday volume vs yesterday's full-day volume (raw burst,
    #    no 10-day smoothing — independent of the score's RVOL term)
    if intra_volume_today is not None and hist_volume is not None and len(hist_volume) >= 2:
        try:
            yday_vol = float(hist_volume.iloc[-2])
        except Exception:
            yday_vol = 0.0
        if yday_vol > 0:
            burst = intra_volume_today / yday_vol
            if burst > 1.5:
                score += 15
            elif burst > 1.0:
                score += 8
            elif burst < 0.5:
                score -= 8

    # 3. Currently holding above intraday VWAP
    if vwap_dist is not None:
        if vwap_dist > 1:
            score += 10
        elif vwap_dist > -0.5:
            score += 5
        else:
            score -= 10

    return round(max(0.0, min(100.0, score)), 1)


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

        # Latest intraday close — used as the "current price" reference for
        # the VWAP comparison below. Defaults to the daily close so the
        # downstream code still has a value when intraday data is missing.
        last_intraday_close = price

        if intra is not None and not intra.empty and len(intra) >= 3:
            iclose = intra["Close"]
            iopen = intra["Open"]
            ivol = intra["Volume"]

            live_intraday_volume = float(ivol.sum())

            last_close = float(iclose.iloc[-1])
            last_intraday_close = last_close
            prev_close_intraday = float(iclose.iloc[-2])
            if prev_close_intraday > 0:
                premarket_pct = (last_close - prev_close_intraday) / prev_close_intraday * 100

            typical_price = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            total_vol = ivol.sum()
            if total_vol > 0:
                vwap_val = float((typical_price * ivol).sum() / total_vol)
                if vwap_val > 0:
                    # Compare the latest intraday tick (not yesterday's daily
                    # close) to today's intraday VWAP.
                    vwap_dist = (last_close - vwap_val) / vwap_val * 100

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

        # ATR — Average True Range, used downstream for volatility-scaled
        # targets/stops. Uses up to 7 daily periods (we only have ~10 to
        # begin with). Falls back to None if data is too thin.
        atr = None
        if len(hist) >= 3:
            high = hist["High"]
            low = hist["Low"]
            prev_close_series = hist["Close"].shift(1)
            tr = pd.concat(
                [
                    high - low,
                    (high - prev_close_series).abs(),
                    (low - prev_close_series).abs(),
                ],
                axis=1,
            ).max(axis=1)
            n_atr = min(7, len(tr.dropna()))
            if n_atr >= 2:
                atr_val = float(tr.dropna().iloc[-n_atr:].mean())
                if atr_val > 0:
                    atr = atr_val

        if enable_ofb_filter:
            if order_flow_bias is None or order_flow_bias < min_ofb:
                return None

        squeeze = False
        low_float = False
        catalyst = False
        sector = "Unknown"
        industry = "Unknown"
        short_pct_display = None
        sentiment_score_val = 0.0  # will be overridden by Finviz sentiment later

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
                # Yahoo news only used to set generic 'catalyst' flag;
                # sentiment is now Finviz-only.
                news = stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub = datetime.fromtimestamp(news[0]["providerPublishTime"], tz=timezone.utc)
                    catalyst = (datetime.now(timezone.utc) - pub).days <= 3
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
        mom_idx = momentum_index(score, rvol10, premarket_pct, m10)

        entry_conf = entry_confidence_score(vwap_dist, rvol10, order_flow_bias)
        bci = breakout_confirmation_index(
            last_price=last_intraday_close,
            hist_close=close,
            hist_volume=daily_volume,
            intra_volume_today=live_intraday_volume,
            vwap_dist=vwap_dist,
        )

        # AI commentary text here still uses 0 sentiment; we will recompute with
        # Finviz sentiment in the display layer.
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
            "Momentum_Index": mom_idx,
            "ATR": round(atr, 4) if atr is not None else None,
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


def browser_notification_component(pending_alerts: list[dict]) -> str:
    """HTML+JS for an iframe component that:
      1. Renders a one-click button to grant Notification permission
      2. Shows current permission status
      3. Fires native OS notifications for any pending alerts in this scan,
         deduped per-symbol-per-calendar-day via localStorage so a stuck
         alert doesn't re-fire on every auto-refresh.

    Uses st.components.v1.html (an iframe), not st.markdown — Streamlit
    strips <script> from markdown for security, so the JS would not run.
    """
    # Defensive against unusual chars in scraped data, though stock
    # tickers shouldn't contain </script> in practice.
    alerts_json = json.dumps(pending_alerts).replace("</", "<\\/")
    return f"""
    <div style="display:flex;align-items:center;gap:10px;
                padding:8px 12px;background:#f7f7f9;border-radius:6px;
                font-family:system-ui,-apple-system,sans-serif;font-size:13px;
                border:1px solid #e5e5e9;">
      <button id="ml-notif-btn"
              style="padding:6px 12px;border-radius:4px;cursor:pointer;
                     border:1px solid #888;background:#fff;font-size:13px;">
        🔔 Enable Browser Notifications
      </button>
      <span id="ml-notif-status" style="color:#666;"></span>
    </div>
    <script>
    (function() {{
      const PENDING = {alerts_json};
      const btn = document.getElementById('ml-notif-btn');
      const stat = document.getElementById('ml-notif-status');

      function refresh() {{
        if (!('Notification' in window)) {{
          stat.textContent = 'not supported in this browser';
          btn.disabled = true;
          return;
        }}
        const p = Notification.permission;
        if (p === 'granted') {{
          stat.textContent = '✓ enabled — alerts go to your OS notification center';
          stat.style.color = '#2a7';
          btn.textContent = '🔔 Notifications enabled';
          btn.style.background = '#e8f7ee';
        }} else if (p === 'denied') {{
          stat.textContent = '✗ blocked by browser — re-enable in site settings';
          stat.style.color = '#c33';
        }} else {{
          stat.textContent = 'click once to enable (browser will ask)';
        }}
      }}
      refresh();

      btn.addEventListener('click', function() {{
        Notification.requestPermission().then(function(p) {{
          refresh();
          if (p === 'granted') {{
            new Notification('ML Screener', {{
              body: "You'll get a ping here when symbols cross your alert thresholds.",
            }});
          }}
        }});
      }});

      if ('Notification' in window && Notification.permission === 'granted') {{
        // Per-calendar-day dedup so reopening the tab during the day
        // doesn't re-fire every notification on the first scan.
        let fired;
        try {{
          fired = JSON.parse(localStorage.getItem('mlscreener_fired') || '{{}}');
        }} catch(e) {{ fired = {{}}; }}
        const today = new Date().toISOString().slice(0, 10);
        if (fired.date !== today) {{
          fired = {{date: today, symbols: {{}}}};
        }}
        for (const a of PENDING) {{
          if (!fired.symbols[a.symbol]) {{
            new Notification('🔥 ' + a.symbol + ' — ML Screener', {{
              body: a.reason,
              tag: 'mlscreener-' + a.symbol,
              requireInteraction: false,
            }});
            fired.symbols[a.symbol] = Date.now();
          }}
        }}
        localStorage.setItem('mlscreener_fired', JSON.stringify(fired));
      }}
    }})();
    </script>
    """


# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning (10-day momentum, V12 hybrid universe)…"):
    if use_last_results and "last_df" in st.session_state:
        df_raw = st.session_state["last_df"].copy()
    else:
        effective_universe_mode = universe_mode
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
    # --- FINVIZ PER-TICKER NEWS + AUTO-SEED ---
    finviz_cache: dict[str, list[dict]] = {}
    now_utc = datetime.now(timezone.utc)
    now_utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    breaking_cutoff = now_utc - timedelta(seconds=BREAKING_WINDOW_SECONDS)
    is_warmup_pass = not st.session_state.finviz_warmup_done

    seed_map = {entry["Symbol"]: entry for entry in st.session_state.seed_universe}

    for sym in df_raw["Symbol"].unique():
        try:
            items = get_finviz_news_for_ticker(sym)
        except Exception:
            items = []

        # Mark each item's `breaking` flag based on when THIS session first
        # observed the URL — not the headline's publish timestamp. Fixes the
        # "stale Friday headline shows as <20m old on Monday" problem.
        for item in items:
            url = item.get("url")
            if not url:
                item["breaking"] = False
                continue
            first_seen = st.session_state.first_seen_finviz_urls.get(url)
            if first_seen is None:
                if is_warmup_pass:
                    # Seed history without firing badges on the first pass.
                    st.session_state.first_seen_finviz_urls[url] = breaking_cutoff
                    item["breaking"] = False
                else:
                    st.session_state.first_seen_finviz_urls[url] = now_utc
                    item["breaking"] = True
            else:
                item["breaking"] = first_seen > breaking_cutoff

        finviz_cache[sym] = items

        if items:
            # auto-seed when Finviz news exists today
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
    st.session_state.finviz_warmup_done = True

    # Finviz presence, seed timestamp, and Finviz-based sentiment
    df_raw["FinvizNews"] = df_raw["Symbol"].map(lambda s: bool(finviz_cache.get(s)))
    df_raw["LastNewsSeed"] = df_raw["Symbol"].map(
        lambda s: seed_map.get(s, {}).get("LastNewsSeed")
    )
    df_raw["FinvizSentiment"] = df_raw["Symbol"].map(
        lambda s: finviz_sentiment_from_items(finviz_cache.get(s, []))
    )
    # Overwrite Sentiment column with Finviz-only sentiment for downstream use
    df_raw["Sentiment"] = df_raw["FinvizSentiment"]

    # --- APPLY FILTERS ---
    us_exchanges = ["NASDAQ", "NYSE", "AMEX", "NYSE/AMEX/ARCA"]

    if catalyst_finviz_only:
        # STRICT for US: require Finviz news (today only)
        # Canadian tickers are still allowed through even without Finviz.
        df = df_raw[
            (df_raw["FinvizNews"])
            | (~df_raw["Exchange"].isin(us_exchanges))
        ].copy()
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
        # Sort by breakout score, then Finviz presence, then PM%, then RSI
        df = df.sort_values(
            by=["Score", "FinvizNews", "PM%", "RSI7"],
            ascending=[False, False, False, False],
        )

        st.subheader(f"🔥 10-Day Momentum Board (V12) — {len(df)} symbols")

        # Pre-scan: collect symbols crossing alert thresholds this scan that
        # haven't already fired in this Python session. This list drives the
        # browser-notification component. The inline row loop below still
        # fires the audio + yellow banner alerts (additive, not replacement).
        pending_browser_alerts: list[dict] = []
        if enable_alerts:
            for _, row in df.iterrows():
                sym = row["Symbol"]
                if sym in st.session_state.alerted:
                    continue
                reason = None
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    reason = f"Score {row['Score']}"
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    reason = f"Premarket {row['PM%']}%"
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    reason = f"VWAP Dist {row['VWAP%']}%"
                if reason:
                    pending_browser_alerts.append({"symbol": sym, "reason": reason})

        # Always render the component (even with no pending alerts) so the
        # permission button stays visible and the user can grant access.
        components.html(
            browser_notification_component(pending_browser_alerts),
            height=60,
        )

        if enable_alerts and st.session_state.alerted:
            alerted_list = ", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        for _, row in df.iterrows():
            sym = row["Symbol"]
            finviz_items = finviz_cache.get(sym, [])
            has_finviz = bool(finviz_items)
            has_breaking = any(n.get("breaking") for n in finviz_items)
            last_news_seed = row.get("LastNewsSeed", None)
            exch_str = str(row.get("Exchange", "") or "")

            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym, f"Score {row['Score']}")
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym, f"Premarket {row['PM%']}%")
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym, f"VWAP Dist {row['VWAP%']}%")

            c1, c2, c3, c4 = st.columns([2, 3, 3, 3])

            c1.markdown(f"**{sym}** ({row['Exchange']})")

            if has_finviz and has_breaking:
                c1.markdown("🆕 **NEW Finviz Headline (just landed)**")
            elif has_finviz:
                c1.markdown("🔥 **Finviz Catalyst (Today)**")
            else:
                # Distinguish CAD vs US when Finviz is unavailable
                if exch_str.upper().startswith("TSX"):
                    c1.markdown("⚪ Finviz not available for CAD ticker")
                else:
                    c1.markdown("⚪ No Finviz Catalyst Today")

            c1.write(f"💲 Price: {row['Price']}")
            c1.write(f"📊 Live Volume: {row['Volume']:,}")
            c1.write(f"🔥 Score: **{row['Score']}**")
            c1.write(f"📊 Momentum Index: {row['Momentum_Index']}/100")
            c1.write(f"{row['MTF_Trend']}")
            c1.write(f"Trend: {row['EMA10 Trend']}")

            if last_news_seed:
                c1.caption(f"Last News Seeded: {last_news_seed}")
            else:
                c1.caption("Last News Seeded: —")

            price_val = float(row["Price"])
            bci_val = row.get("Breakout_Confirm", 0.0)
            entry_val = row.get("Entry_Confidence", 0.0)
            atr_val = row.get("ATR", None)

            try:
                if bci_val is None or pd.isna(bci_val):
                    bci_val = 0.0
                if entry_val is None or pd.isna(entry_val):
                    entry_val = 0.0
            except Exception:
                bci_val = bci_val or 0.0
                entry_val = entry_val or 0.0

            try:
                if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
                    atr_val = None
            except Exception:
                atr_val = None

            if atr_val is not None:
                # ATR-scaled target/stop. Wider target with stronger
                # breakout confirmation; tighter stop with higher entry
                # confidence. Multipliers stay bounded so a high-confidence
                # trade can't end up with a zero-distance stop.
                target_mult = 1.5 + (bci_val / 100.0)            # 1.5x to 2.5x ATR
                stop_mult = 1.5 - 0.5 * (entry_val / 100.0)      # 1.0x to 1.5x ATR
                target_price = round(price_val + target_mult * atr_val, 2)
                stop_price = round(max(price_val - stop_mult * atr_val, 0.01), 2)

                c1.write(f"🎯 Target: **${target_price}** (≈ {target_mult:.2f}× ATR)")
                c1.write(f"🛑 Stop: **${stop_price}** (≈ {stop_mult:.2f}× ATR)")

                try:
                    rr = (target_price - price_val) / max(0.01, (price_val - stop_price))
                    rr_text = f"{rr:.2f} : 1"
                except Exception:
                    rr_text = "—"
                c1.write(f"📈 R:R: **{rr_text}**")
                c1.caption(f"ATR(≤7d) = {atr_val:.3f}")
            else:
                c1.write("🎯 Target: — (insufficient data for ATR)")
                c1.write("🛑 Stop: —")
                c1.write("📈 R:R: —")

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

            target_rationale = " ".join(ai_expl_list)
            c1.markdown(f"🧠 **Target Rationale:** {target_rationale}")

            c2.write(f"PM%: {row['PM%']}")
            c2.write(f"YDay%: {row['YDay%']}")
            c2.write(f"3D%: {row['3D%']}  |  10D%: {row['10D%']}")
            c2.write(f"RSI7: {row['RSI7']}  |  RVOL_10D: {row['RVOL_10D']}x")
            c2.write(f"Breakout Confirm: {row.get('Breakout_Confirm', 0)} / 100")
            c2.write(f"Entry Confidence: {row.get('Entry_Confidence', 0)} / 100")

            c3.write(f"VWAP Dist %: {row['VWAP%']}")
            c3.write(f"Order Flow Bias: {row['FlowBias']}")

            # Finviz sentiment numeric + emoji
            sent_val = row.get("FinvizSentiment", 0.0)
            if sent_val > 0.4:
                sent_emoji = "🟢"
            elif sent_val < -0.4:
                sent_emoji = "🔴"
            else:
                sent_emoji = "⚪"

            if enable_enrichment:
                c3.write(
                    f"Squeeze: {row['Squeeze?']} | LowFloat: {row['LowFloat?']}"
                )
                c3.write(f"Sec/Ind: {row['Sector']} / {row['Industry']}")
                c3.write(f"News Sentiment (Finviz): {sent_val:+.2f} {sent_emoji}")
            else:
                c3.write("Enrichment: OFF (float/short/news skipped for speed)")
                c3.write(f"News Sentiment (Finviz): {sent_val:+.2f} {sent_emoji}")

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

            # Recompute AI commentary using Finviz sentiment so it reflects headline tone
            ai_view = ai_commentary(
                score=row["Score"],
                pm=row["PM%"],
                rvol=row["RVOL_10D"],
                flow_bias=row["FlowBias"],
                vwap=row["VWAP%"],
                ten_day=row["10D%"],
                sentiment=sent_val,
                entry_conf=row.get("Entry_Confidence", 0.0),
                bci=row.get("Breakout_Confirm", 0.0),
                preopen_mode=preopen_mode,
            )
            c3.markdown(f"🧠 **AI View:** {ai_view}")

            # --- Ticker-specific Finviz headlines (today-only) with BREAKING badge ---
            with c3.expander("📰 Recent Headlines (Finviz, Today Only)", expanded=True):
                if not finviz_items:
                    st.write("No Finviz headlines today for this ticker.")
                else:
                    for n in finviz_items:
                        badge = "🆕 NEW • " if n.get("breaking") else ""
                        st.markdown(
                            f"{n['sent']} {badge}"
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
                            "Momentum_Index",
                            "PM%",
                            "10D%",
                            "RVOL_10D",
                            "VWAP%",
                            "FlowBias",
                            "Breakout_Confirm",
                            "Entry_Confidence",
                            "ATR",
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
            "Momentum_Index",
            "ATR",
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
            "FinvizSentiment",
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






