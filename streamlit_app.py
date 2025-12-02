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

# ========================= SETTINGS =========================
THREADS               = 20           # keep high but not crazy
AUTO_REFRESH_DEFAULT  = 120_000      # default auto-refresh every 120 seconds
HISTORY_LOOKBACK_DAYS = 10           # 🔥 10-day mode
INTRADAY_INTERVAL     = "2m"         # 2-minute candles
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= SESSION STATE FOR V11 STREAMING =========================
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "auto_refresh_ms" not in st.session_state:
    st.session_state.auto_refresh_ms = AUTO_REFRESH_DEFAULT

# ========================= AUTO REFRESH (V11 streaming aware) =========================
if st.session_state.auto_refresh_enabled:
    st_autorefresh(interval=st.session_state.auto_refresh_ms, key="refresh_v11")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V11 – 10-Day Momentum Screener (Hybrid Volume/Randomized + ML/AI)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V11 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized + ML/AI)")
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Audio alerts • V9/V10/V11 universe modes "
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
        min_value=50,
        max_value=600,
        value=2000,
        step=50,
        help="Keeps scans fast when you don't use a custom watchlist.",
    )

    # 🔥 V9 universe mode
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
            "Classic = original V8 behavior.\n"
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
    # V10 requirement: min volume can go down to 0, default unchanged
    min_volume = st.number_input("Min Daily Volume", 0, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 1.0)

    # ✅ NEW: Min Breakout Confirmation & Entry Confidence filters (directly under breakout)
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
    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    # 🔁 V9: optional order-flow bias filter
    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox(
        "Use Min Order Flow Bias Filter",
        value=False,
        help="When enabled, only keep symbols where buy volume dominates."
    )
    min_ofb = st.slider(
        "Min Order Flow Bias (0–1, buyer control)",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="0.5 = equal buy/sell; 0.7 = strong buyer control."
    )

    # V11: Ignore filters when watchlist populated (watchlist precedence)
    ignore_filters_for_watchlist = st.checkbox(
        "Ignore filters when watchlist is populated (V11)",
        value=False,
        help="When enabled and watchlist has symbols, hard filters (price, volume, etc.) are skipped."
    )

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")

    # 🔇 V10+V11: alerts default to disabled
    enable_alerts = st.checkbox(
        "Enable Audio + Alert Banner",
        value=False,
        help="Turn this off to completely silence alerts."
    )

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    st.markdown("---")
    # V11 streaming controls
    st.subheader("V11 Streaming Controls")
    auto_refresh_enabled = st.checkbox(
        "Enable Auto-Refresh (Streaming)",
        value=st.session_state.auto_refresh_enabled,
        help="Controls whether the app auto-refreshes. Takes effect on next refresh."
    )
    auto_refresh_ms = st.slider(
        "Auto-Refresh Interval (ms)",
        min_value=10_000,
        max_value=300_000,
        value=st.session_state.auto_refresh_ms,
        step=5_000,
        help="Used when auto-refresh is enabled."
    )
    st.session_state.auto_refresh_enabled = auto_refresh_enabled
    st.session_state.auto_refresh_ms = auto_refresh_ms

    # V11: pre-open scanning mode
    preopen_mode = st.checkbox(
        "Pre-Open Scan Mode (V11)",
        value=False,
        help="Emphasize premarket moves & volume; de-emphasize longer-term trend."
    )

    # V11: universe caching persistence
    use_last_results = st.checkbox(
        "Use last scan results (no rescan, V11)",
        value=False,
        help="Use cached universe from prior run instead of rescanning."
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

    # Robust symbol extraction: use first column if "Symbol" not present
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
    - Classic alphabetical slice
    - Randomized slice
    - Live volume-ranked (V9)
    """
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n", " ").replace(",", " ").split()
        tickers = sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol": t, "Exchange": "WATCH"} for t in tickers]

    syms = load_symbols()

    # V9 modes
    if universe_mode == "Randomized Slice":
        base = syms[:]
        random.shuffle(base)
        return base[:max_universe]

    if universe_mode == "Live Volume Ranked (slower)":
        base = syms[:volume_rank_pool]
        ranked = []
        for sym in base:
            try:
                t = yf.Ticker(sym["Symbol"])
                d = t.history(period="1d", interval=INTRADAY_INTERVAL, prepost=True)
                if not d.empty:
                    # Live volume = cumulative intraday volume
                    live_vol = float(d["Volume"].sum())
                    ranked.append({**sym, "LiveVol": live_vol})
            except Exception:
                continue

        if not ranked:
            # fallback to classic if volume fetch fails
            return syms[:max_universe]

        ranked_sorted = sorted(
            ranked,
            key=lambda x: x.get("LiveVol", 0.0),
            reverse=True
        )
        return ranked_sorted[:max_universe]

    # Classic (V8 behavior)
    return syms[:max_universe]


# ========================= SCORING (10-DAY MODEL) =========================
def short_window_score(pm, yday, m3, m10, rsi7, rvol10, catalyst, squeeze, vwap, flow_bias, preopen_mode=False):
    """
    pm        = premarket % (2m candles)
    yday      = yesterday % move
    m3        = 3-day % move
    m10       = 10-day % move
    rsi7      = RSI(7)
    rvol10    = relative volume vs 10-day avg
    catalyst  = bool
    squeeze   = bool
    vwap      = % above VWAP
    flow_bias = 0..1 (buy volume / total volume)
    preopen_mode = if True, emphasize premarket and volume, de-emphasize 10D trend
    """
    score = 0.0

    # Pre-open mode tweaks weights a bit toward PM & RVOL
    pm_w   = 2.0 if preopen_mode else 1.6
    m10_w  = 0.3 if preopen_mode else 0.6
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

    if vwap is not None and vwap > 0:
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
    V11: 'ML-style' probability-like number, using a richer feature mix
    but kept lightweight (no external libraries).
    """
    try:
        # feature engineering-ish
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
    V11: very lightweight sentiment scorer using keywords.
    Returns value in roughly [-1, 1].
    """
    text = (title or "") + " " + (summary or "")
    text = text.lower()

    pos_words = [
        "beat", "beats", "strong", "surge", "upgrade", "upgrades", "bullish",
        "raises", "raise", "record", "jump", "rally", "soars", "soar", "momentum"
    ]
    neg_words = [
        "miss", "misses", "weak", "downgrade", "downgrades", "bearish",
        "cuts", "cut", "plunge", "fall", "falls", "tumbles", "tumble",
        "guidance cut", "warning"
    ]

    score = 0
    for w in pos_words:
        if w in text:
            score += 1
    for w in neg_words:
        if w in text:
            # negative words subtract
            score -= 1

    # squash to [-1,1]
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 5.0))


def entry_confidence_score(vwap_dist, rvol10, flow_bias) -> float:
    """
    V11: Entry timing confidence 0–100 based on VWAP distance, RVOL, and OFB.
    Idea: modestly above VWAP, good RVOL, buy-dominant order flow → higher score.
    """
    if vwap_dist is None or rvol10 is None or flow_bias is None:
        return 50.0  # neutral

    score = 60.0

    # ideal VWAP zone: 0 to +3%
    if -1 <= vwap_dist <= 3:
        score += 15
    elif abs(vwap_dist) > 8:
        score -= 15

    # RVOL contribution
    if rvol10 > 2:
        score += 10
    elif rvol10 < 0.7:
        score -= 10

    # order flow bias
    score += (flow_bias - 0.5) * 40.0

    return round(max(0.0, min(100.0, score)), 1)


def breakout_confirmation_index(score, rvol10, pm, m10) -> float:
    """
    V11: Breakout confirmation index 0–100 combining score, RVOL, PM, and 10D trend.
    """
    base = (score or 0) / 2.0  # 0–100 if score ~ 0–200
    if rvol10 is not None:
        base += max(0.0, (rvol10 - 1.0) * 8.0)
    if pm is not None:
        base += max(0.0, pm) * 1.2
    if m10 is not None and m10 > 0:
        base += min(m10, 30) * 0.8

    return round(max(0.0, min(100.0, base)), 1)


# ========================= SIMPLE AI COMMENTARY (V11 upgraded) =========================
def ai_commentary(score, pm, rvol, flow_bias, vwap, ten_day, sentiment, entry_conf, bci, preopen_mode):
    comments = []

    # High level regime
    if score is not None:
        if score >= 90:
            comments.append("Explosive momentum profile, risk-on candidate.")
        elif score >= 60:
            comments.append("Constructive momentum with improving structure.")
        elif score >= 30:
            comments.append("Early momentum, still needs confirmation.")

    # Premarket behavior
    if pm is not None:
        if pm > 5:
            comments.append("Strong premarket demand showing early accumulation.")
        elif pm < -3:
            comments.append("Notable premarket supply; caution on chasing intraday pops.")

    # Volume / liquidity
    if rvol is not None:
        if rvol > 2:
            comments.append("Volume aggressively expanding vs 10-day baseline.")
        elif rvol < 0.7:
            comments.append("Liquidity muted; slippage/whipsaws more likely.")

    # Order flow
    if flow_bias is not None:
        if flow_bias > 0.7:
            comments.append("Buyers dominating tape, dips may get absorbed quickly.")
        elif flow_bias < 0.4:
            comments.append("Sellers pressing, rallies could be sold into.")

    # VWAP / positioning
    if vwap is not None:
        if 0 <= vwap <= 3:
            comments.append("Trading near/just above VWAP – healthy risk/reward zone.")
        elif vwap > 5:
            comments.append("Extended well above VWAP – momentum but risk of chase.")
        elif vwap < 0:
            comments.append("Below VWAP – still under distribution pressure.")

    # 10-day structure
    if ten_day is not None:
        if ten_day > 15:
            comments.append("10D structure confirmed uptrend; pullbacks may be buyable.")
        elif ten_day < -8:
            comments.append("10D trend in clear distribution – countertrend risk.")

    # News sentiment
    if sentiment is not None:
        if sentiment > 0.4:
            comments.append("Headline flow skewed positive; narrative supportive.")
        elif sentiment < -0.4:
            comments.append("Recent headlines skewed negative; narrative drag present.")

    # Pre-open mode note
    if preopen_mode:
        comments.append("Pre-open mode: signal weights biased toward PM and early volume.")

    # Entry confidence & BCI
    comments.append(f"Entry confidence ~ {entry_conf:.0f}/100.")
    comments.append(f"Breakout confirmation ~ {bci:.0f}/100.")

    # Fallback
    if not comments:
        return "Neutral / indecisive tape — watching for clearer confirmation."

    return " | ".join(comments)


# ========================= CORE SCAN =========================
def scan_one(sym, enable_enrichment: bool, enable_ofb_filter: bool, min_ofb: float, preopen_mode: bool):
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

        # Use *last close* as anchor price
        price = float(close.iloc[-1])
        daily_vol_last = float(daily_volume.iloc[-1])

        # Intraday 2m history for PM, VWAP, order flow, LIVE VOLUME
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

            # LIVE VOLUME REPLACEMENT: use cumulative intraday volume
            live_intraday_volume = float(ivol.sum())

            # Premarket % from last two bars
            last_close = float(iclose.iloc[-1])
            prev_close_intraday = float(iclose.iloc[-2])
            if prev_close_intraday > 0:
                premarket_pct = (last_close - prev_close_intraday) / prev_close_intraday * 100

            # VWAP
            typical_price = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            total_vol = ivol.sum()
            if total_vol > 0:
                vwap_val = float((typical_price * ivol).sum() / total_vol)
                if vwap_val > 0:
                    vwap_dist = (price - vwap_val) / vwap_val * 100

            # Order-flow bias: buy vs sell volume
            of_df = intra[["Open", "Close", "Volume"]].dropna()
            if not of_df.empty:
                sign = (of_df["Close"] > of_df["Open"]).astype(int) - (of_df["Close"] < of_df["Open"]).astype(int)
                buy_vol = float((of_df["Volume"] * (sign > 0)).sum())
                sell_vol = float((of_df["Volume"] * (sign < 0)).sum())
                total = buy_vol + sell_vol
                if total > 0:
                    order_flow_bias = buy_vol / total  # 0..1
               
        # --- FIX: True premarket price override when 2m bars are missing ---
        try:
            fi = stock.fast_info
            pre_price = fi.get("last_price", None)
            prev_close = fi.get("regular_market_previous_close", None)

            if pre_price and prev_close and prev_close > 0:
                calc_pm = (pre_price - prev_close) / prev_close * 100

                # Only override during premarket hours (before 9:30am ET)
                now = datetime.now(timezone.utc)
                if now.hour < 14 or (now.hour == 14 and now.minute < 30):
                    premarket_pct = round(calc_pm, 2)
        except Exception:
            pass

        # If we didn't get live intraday volume, fall back to daily
        if live_intraday_volume is None:
            live_intraday_volume = daily_vol_last

        # Price/volume filters (may be ignored for watchlist via caller logic)
        if price > max_price or live_intraday_volume < min_volume:
            return None

        # Momentum windows: yesterday, 3-day, 10-day
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

        # RSI(7)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi7 = float(rsi_series.iloc[-1])

        # EMA10
        ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
        ema_trend = "🔥 Breakout" if price > ema10 and rsi7 > 55 else "Neutral"

        # 10-day RVOL using LIVE VOLUME vs average daily volume
        avg10 = float(daily_volume.mean()) if len(daily_volume) > 0 else 0
        rvol10 = live_intraday_volume / avg10 if avg10 > 0 else None

        # Optional order-flow bias filter
        if enable_ofb_filter:
            if order_flow_bias is None or order_flow_bias < min_ofb:
                return None

        # Enrichment (float, short, news, sentiment)
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

               # V11: sentiment scoring from multiple recent news items
                sent_vals = []

                for n in news[:5]:  # analyze up to 5 recent articles
                    t = n.get("title", "")
                    s = n.get("summary", "")
                    sent_vals.append(news_sentiment_score(t, s))

                if sent_vals:
                    sentiment_score_val = round(sum(sent_vals) / len(sent_vals), 2)
                else:
                    sentiment_score_val = 0.0

            except Exception:
                pass

        # Multi-timeframe label
        mtf_label = multi_timeframe_label(premarket_pct, m3, m10)

        # Score + ML-style probability
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
        )
        prob_rise = ml_breakout_probability(score, rvol10, premarket_pct, m10)

        # V11: entry timing & breakout confirmation index
        entry_conf = entry_confidence_score(vwap_dist, rvol10, order_flow_bias)
        bci = breakout_confirmation_index(score, rvol10, premarket_pct, m10)

        # AI commentary (upgraded)
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

        # Sparkline: 10d closes
        spark_series = close

        return {
            "Symbol": ticker,
            "Exchange": exchange,
            "Price": round(price, 2),
            "Volume": int(live_intraday_volume),     # 🔥 live intraday volume
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
):
    """
    V11 lightning engine:
    - parallel scan via ThreadPool
    - universe constructed by build_universe
    - optional ignoring of filters for watchlist handled by wrapper logic
    """
    universe = build_universe(
        watchlist_text,
        max_universe,
        universe_mode,
        volume_rank_pool,
    )
    results = []

    # If ignoring filters for watchlist and watchlist is populated,
    # we temporarily relax min_volume & max_price via globals-like hack
    global min_volume, max_price
    saved_min_volume = min_volume
    saved_max_price = max_price

    if ignore_filters_for_watchlist_flag and watchlist_text.strip():
        min_volume = 0
        max_price = 10_000.0  # effectively disable

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
                )
                for sym in universe
            ]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res:
                    results.append(res)
    finally:
        # restore filters for rest of app
        min_volume = saved_min_volume
        max_price = saved_max_price

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ========================= SPARKLINE & CHART HELPERS =========================
def sparkline(series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
    ))
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
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines+markers",
        name=title,
    ))
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
with st.spinner("Scanning (10-day momentum, V11 hybrid universe)…"):
    if use_last_results and "last_df" in st.session_state:
        df = st.session_state["last_df"].copy()
    else:
        df = run_scan(
            watchlist_text,
            max_universe,
            enable_enrichment,
            enable_ofb_filter,
            min_ofb,
            universe_mode,
            volume_rank_pool,
            preopen_mode,
            ignore_filters_for_watchlist,
        )
        st.session_state["last_df"] = df.copy()

if df.empty:
    st.error("No results found. Try adding a watchlist or relaxing filters.")
else:
    # Apply filters (unless ignore_filters_for_watchlist is on with non-empty watchlist)
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

        # ✅ NEW FILTERS: Breakout_Confirm & Entry_Confidence
        if min_breakout_confirm > 0.0 and "Breakout_Confirm" in df.columns:
            df = df[df["Breakout_Confirm"].fillna(-1) >= min_breakout_confirm]

        if min_entry_confidence > 0.0 and "Entry_Confidence" in df.columns:
            df = df[df["Entry_Confidence"].fillna(-1) >= min_entry_confidence]

    if df.empty:
        st.error("No results left after filters. Try relaxing constraints or disabling 'Ignore filters' toggle.")
    else:
        df = df.sort_values(by=["Score", "PM%", "RSI7"], ascending=[False, False, False])

        st.subheader(f"🔥 10-Day Momentum Board (V11) — {len(df)} symbols")

        # 🔔 Alert banner (V9+V10+V11)
        if enable_alerts and st.session_state.alerted:
            alerted_list = ", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        # Iterate + audio alerts + inline charts
        for _, row in df.iterrows():
            sym = row["Symbol"]

            # Audio alerts (once per symbol, if enabled)
            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym, f"Score {row['Score']}")
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym, f"Premarket {row['PM%']}%")
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym, f"VWAP Dist {row['VWAP%']}%")

            c1, c2, c3, c4 = st.columns([2, 3, 3, 3])

            # Column 1: Basic info + score + volume
            c1.markdown(f"**{sym}** ({row['Exchange']})")
            c1.write(f"💲 Price: {row['Price']}")
            c1.write(f"📊 Live Volume: {row['Volume']:,}")
            c1.write(f"🔥 Score: **{row['Score']}**")
            c1.write(f"🤖 ML Prob_Rise: {row['Prob_Rise%']}%")
            c1.write(f"{row['MTF_Trend']}")
            c1.write(f"Trend: {row['EMA10 Trend']}")

            # 🎯 AI Target & 🛑 Stop (simple heuristic, not advice)
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
            ai_stop   = round(price_val * (1 - (1 - entry_val / 100.0) * 0.05), 2)

            c1.write(f"🎯 AI Target: **${ai_target}**")
            c1.write(f"🛑 AI Stop: **${ai_stop}**")
           
            try:
                rr = (ai_target - price_val) / max(0.01, (price_val - ai_stop))
                rr_text = f"{rr:.2f} : 1"
            except Exception:
                rr = None
                rr_text = "—"
            
            c1.write(f"📈 R:R: **{rr_text}**")
            
            # --- AI Explanation for Targets/Stops ---
            ai_expl_list = []
            
            # Breakout confirmation
            if bci_val >= 70:
                ai_expl_list.append("Breakout structure strongly confirmed.")
            elif bci_val >= 50:
                ai_expl_list.append("Moderate breakout confirmation present.")
            else:
                ai_expl_list.append("Weak confirmation — target conservative.")
            
            # Entry timing
            if entry_val >= 70:
                ai_expl_list.append("Entry confidence high; tape favoring long entries.")
            elif entry_val >= 50:
                ai_expl_list.append("Entry timing acceptable.")
            else:
                ai_expl_list.append("Entry window uncertain; volatility elevated.")
            
            # VWAP positioning
            if row["VWAP%"] is not None:
                if row["VWAP%"] > 0:
                    ai_expl_list.append("Price holding above VWAP (bullish positioning).")
                else:
                    ai_expl_list.append("Below VWAP — higher risk of failed breakout.")
            
            # 10-day trend
            if row["10D%"] is not None:
                if row["10D%"] > 0:
                    ai_expl_list.append("10-day trend supportive.")
                else:
                    ai_expl_list.append("10-day trend weak — target reduced.")
            
            # Order Flow Bias
            flow = row.get("FlowBias", None)
            if flow is not None:
                if flow > 0.6:
                    ai_expl_list.append("Buyers absorbing dips; strong participation.")
                elif flow < 0.4:
                    ai_expl_list.append("Sellers active — cautious stop placement.")
            
            # Final AI narrative
            ai_target_expl = " ".join(ai_expl_list)
            c1.markdown(f"🧠 **AI Target Rationale:** {ai_target_expl}")


            # Column 2: Momentum snapshot + confirmation
            c2.write(f"PM%: {row['PM%']}")
            c2.write(f"YDay%: {row['YDay%']}")
            c2.write(f"3D%: {row['3D%']}  |  10D%: {row['10D%']}")
            c2.write(f"RSI7: {row['RSI7']}  |  RVOL_10D: {row['RVOL_10D']}x")
            c2.write(f"Breakout Confirm: {row.get('Breakout_Confirm', 0)} / 100")
            c2.write(f"Entry Confidence: {row.get('Entry_Confidence', 0)} / 100")

            # Column 3: Microstructure + AI commentary
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

            # AI commentary line
            c3.markdown(f"🧠 **AI View:** {row.get('AI_Commentary', '—')}")

            # Column 4: Sparkline + full chart
            c4.plotly_chart(sparkline(row["Spark"]), use_container_width=False)
            with c4.expander("📊 View 10-day chart"):
                c4.plotly_chart(bigline(row["Spark"], f"{sym} - Last 10 Days"), use_container_width=True)

            st.divider()

        # V11: Watchlist multi-view panels
        raw_watch = watchlist_text.strip()
        if raw_watch:
            raw = raw_watch.replace("\n", " ").replace(",", " ").split()
            wl_tickers = sorted(set(s.upper() for s in raw if s.strip()))
            wl_df = df[df["Symbol"].isin(wl_tickers)]

            if not wl_df.empty:
                st.subheader("📋 Watchlist Multi-View (V11)")
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

        # Download current table
        csv_cols = [
            "Symbol", "Exchange", "Price", "Volume", "Score", "Prob_Rise%",
            "PM%", "YDay%", "3D%", "10D%", "RSI7", "EMA10 Trend",
            "RVOL_10D", "VWAP%", "FlowBias", "Squeeze?", "LowFloat?",
            "Short % Float", "Sector", "Industry", "Catalyst", "MTF_Trend",
            "AI_Commentary", "Sentiment", "Entry_Confidence", "Breakout_Confirm",
        ]
        csv_cols = [c for c in csv_cols if c in df.columns]

        st.download_button(
            "📥 Download Screener CSV",
            data=df[csv_cols].to_csv(index=False),
            file_name="v11_10day_momentum_screener_hybrid_ml_ai.csv",
            mime="text/csv",
        )

st.caption("For research and education only. Not financial advice.")







