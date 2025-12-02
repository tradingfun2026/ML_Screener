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
THREADS               = 20
AUTO_REFRESH_DEFAULT  = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= SESSION STATE =========================
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "auto_refresh_ms" not in st.session_state:
    st.session_state.auto_refresh_ms = AUTO_REFRESH_DEFAULT

if st.session_state.auto_refresh_enabled:
    st_autorefresh(interval=st.session_state.auto_refresh_ms, key="refresh_v11")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V11 – 10-Day Momentum Screener (Hybrid Volume/Randomized + ML/AI)",
    layout="wide",
)

st.title("🚀 V11 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized + ML/AI)")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers",
        value="",
        height=80,
    )

    max_universe = st.slider("Max symbols (no watchlist)", 50, 600, 2000, 50)

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"]
    )

    volume_rank_pool = st.slider("Volume Rank Pool", 100, 2000, 600, 100)

    enable_enrichment = st.checkbox("Include float/short + news (slower)", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume = st.number_input("Min Daily Volume", 0, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 1.0)

    # ============================================================
    # ✅ NEW FILTERS — ONLY ADDITION REQUESTED
    # ============================================================
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
    # ============================================================

    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---")
    st.subheader("Order Flow Filter")
    enable_ofb_filter = st.checkbox("Use Min Order Flow Bias Filter", value=False)
    min_ofb = st.slider("Min Order Flow Bias", 0.00, 1.00, 0.50, 0.01)

    ignore_filters_for_watchlist = st.checkbox("Ignore filters when watchlist populated", value=False)

    st.markdown("---")
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Audio + Alert Banner", value=False)
    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    st.markdown("---")
    st.subheader("V11 Streaming")
    auto_refresh_enabled = st.checkbox("Enable Auto-Refresh (Streaming)", value=st.session_state.auto_refresh_enabled)
    auto_refresh_ms = st.slider("Auto-Refresh Interval (ms)", 10_000, 300_000, st.session_state.auto_refresh_ms, 5_000)
    st.session_state.auto_refresh_enabled = auto_refresh_enabled
    st.session_state.auto_refresh_ms = auto_refresh_ms

    preopen_mode = st.checkbox("Pre-Open Scan Mode", value=False)
    use_last_results = st.checkbox("Use last scan results", value=False)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        if "last_df" in st.session_state:
            del st.session_state["last_df"]
        st.success("Cache cleared — fresh scan will run now.")

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|", engine="python", skipfooter=1, on_bad_lines="skip"
    )
    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|", engine="python", skipfooter=1, on_bad_lines="skip"
    )

    nas_col = "Symbol" if "Symbol" in nasdaq.columns else nasdaq.columns[0]
    other_col = "ACT Symbol" if "ACT Symbol" in other.columns else other.columns[0]

    nasdaq_df = pd.DataFrame({
        "Symbol": nasdaq[nas_col].astype(str).str.strip(),
        "Exchange": "NASDAQ"
    })

    other_df = pd.DataFrame({
        "Symbol": other[other_col].astype(str).str.strip(),
        "Exchange": other.get("Exchange", pd.Series(["NYSE/AMEX/ARCA"] * len(other)))
    })

    df = pd.concat([nasdaq_df, other_df], ignore_index=True).dropna(subset=["Symbol"])
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")


def build_universe(watchlist_text, max_universe, universe_mode, volume_rank_pool):
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n", " ").replace(",", " ").split()
        tickers = sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol": t, "Exchange": "WATCH"} for t in tickers]

    syms = load_symbols()

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
                d = t.history(period="1d", interval="2m", prepost=True)
                if not d.empty:
                    ranked.append({**sym, "LiveVol": float(d["Volume"].sum())})
            except:
                pass

        if not ranked:
            return syms[:max_universe]

        ranked_sorted = sorted(ranked, key=lambda x: x.get("LiveVol", 0), reverse=True)
        return ranked_sorted[:max_universe]

    return syms[:max_universe]


# ========================= SCORING FUNCTIONS =========================
def short_window_score(pm, yday, m3, m10, rsi7, rvol10, catalyst, squeeze, vwap, flow_bias, preopen_mode=False):
    score = 0.0
    pm_w   = 2.0 if preopen_mode else 1.6
    m10_w  = 0.3 if preopen_mode else 0.6
    rvol_w = 2.6 if preopen_mode else 2.0

    if pm: score += max(pm, 0) * pm_w
    if yday: score += max(yday, 0) * 0.8
    if m3: score += max(m3, 0) * 1.2
    if m10: score += max(m10, 0) * m10_w

    if rsi7 and rsi7 > 55: score += (rsi7 - 55) * 0.4
    if rvol10 and rvol10 > 1.2: score += (rvol10 - 1.2) * rvol_w
    if vwap and vwap > 0: score += min(vwap, 6) * 1.5
    if flow_bias: score += (flow_bias - 0.5) * 22.0

    if catalyst: score += 8
    if squeeze: score += 12
    return round(score, 2)


def ml_breakout_probability(score, rvol10, pm, m10):
    try:
        base = score / 25.0
        if rvol10: base += (rvol10 - 1.0) * 0.15
        if pm: base += (pm / 20.0) * 0.2
        if m10: base += (m10 / 50.0) * 0.1
        return round((1 / (1 + math.exp(-base))) * 100, 1)
    except:
        return None


def multi_timeframe_label(pm, m3, m10):
    bull_pm = pm is not None and pm > 0
    bull_3d = m3 is not None and m3 > 0
    bull_10d = m10 is not None and m10 > 0
    positives = sum([bull_pm, bull_3d, bull_10d])

    if positives == 3: return "✅ Aligned Bullish (Intraday + 3D + 10D)"
    if positives == 2: return "🟢 Leaning Bullish"
    if positives == 1: return "🟡 Mixed"
    return "🔻 Not Aligned"


def news_sentiment_score(title, summary=None):
    text = (title or "") + " " + (summary or "")
    text = text.lower()
    pos = ["beat","strong","surge","upgrade","bullish","record","jump","rally","soars"]
    neg = ["miss","weak","downgrade","bearish","cut","plunge","fall","warning"]
    score = sum(1 for w in pos if w in text) - sum(1 for w in neg if w in text)
    return max(-1, min(1, score / 5.0)) if score != 0 else 0.0


def entry_confidence_score(vwap_dist, rvol10, flow_bias):
    if vwap_dist is None or rvol10 is None or flow_bias is None:
        return 50.0
    score = 60
    if -1 <= vwap_dist <= 3: score += 15
    elif abs(vwap_dist) > 8: score -= 15
    if rvol10 > 2: score += 10
    elif rvol10 < 0.7: score -= 10
    score += (flow_bias - 0.5) * 40
    return round(max(0, min(100, score)), 1)


def breakout_confirmation_index(score, rvol10, pm, m10):
    base = (score or 0) / 2.0
    if rvol10: base += max(0, (rvol10 - 1) * 8)
    if pm: base += max(0, pm) * 1.2
    if m10 and m10 > 0: base += min(m10, 30) * 0.8
    return round(max(0, min(100, base)), 1)

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

    # ----------------------------------------
    # NEW FILTERS (Option A — Number Inputs)
    # ----------------------------------------
    min_bci = st.sidebar.number_input(
        "Min Breakout Confirmation (0–100)",
        0, 100, 0
    )

    min_entry_conf = st.sidebar.number_input(
        "Min Entry Confidence (0–100)",
        0, 100, 0
    )
    # ----------------------------------------

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

        # ----------------------------
        # Apply NEW filters
        # ----------------------------
        if min_bci > 0:
            df = df[df["Breakout_Confirm"].fillna(-1) >= min_bci]

        if min_entry_conf > 0:
            df = df[df["Entry_Confidence"].fillna(-1) >= min_entry_conf]
        # ----------------------------

    if df.empty:
        st.error("No results left after filters. Try relaxing constraints or disabling 'Ignore filters' toggle.")
    else:

        df = df.sort_values(by=["Score", "PM%", "RSI7"], ascending=[False, False, False])

        st.subheader(f"🔥 10-Day Momentum Board (V11) — {len(df)} symbols")

        if enable_alerts and st.session_state.alerted:
            alerted_list = ", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        # Iterate + audio alerts + inline charts
        for _, row in df.iterrows():
            sym = row["Symbol"]

            # Audio alerts
            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym, f"Score {row['Score']}")
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym, f"Premarket {row['PM%']}%")
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym, f"VWAP Dist {row['VWAP%']}%")

            c1, c2, c3, c4 = st.columns([2, 3, 3, 3])

            # Column 1
            c1.markdown(f"**{sym}** ({row['Exchange']})")
            c1.write(f"💲 Price: {row['Price']}")
            c1.write(f"📊 Live Volume: {row['Volume']:,}")
            c1.write(f"🔥 Score: **{row['Score']}**")
            c1.write(f"🤖 ML Prob_Rise: {row['Prob_Rise%']}%")
            c1.write(f"{row['MTF_Trend']}")
            c1.write(f"Trend: {row['EMA10 Trend']}")

            # ----------------------------------------
            # AI TARGETS
            # ----------------------------------------
            price = row["Price"]
            bci = row.get("Breakout_Confirm", 0)
            entry_conf = row.get("Entry_Confidence", 0)

            # Target logic: conservative, not trading advice
            ai_target = round(price * (1 + (bci / 250) + (entry_conf / 400)), 2)
            ai_stop   = round(price * (1 - (1 - entry_conf / 100) * 0.05), 2)

            c1.write(f"🎯 **AI Target:** ${ai_target}")
            c1.write(f"🛑 **AI Stop:** ${ai_stop}")
            # ----------------------------------------

            # Column 2
            c2.write(f"PM%: {row['PM%']}")
            c2.write(f"YDay%: {row['YDay%']}")
            c2.write(f"3D%: {row['3D%']}  |  10D%: {row['10D%']}")
            c2.write(f"RSI7: {row['RSI7']}  |  RVOL_10D: {row['RVOL_10D']}x")
            c2.write(f"Breakout Confirm: {row.get('Breakout_Confirm', 0)} / 100")
            c2.write(f"Entry Confidence: {row.get('Entry_Confidence', 0)} / 100")

            # Column 3
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

            # AI commentary
            c3.markdown(f"🧠 **AI View:** {row.get('AI_Commentary', '—')}")

            # Column 4
            c4.plotly_chart(sparkline(row["Spark"]), use_container_width=False)
            with c4.expander("📊 View 10-day chart"):
                c4.plotly_chart(bigline(row["Spark"], f"{sym} - Last 10 Days"), use_container_width=True)

            st.divider()

        # Watchlist multi-view
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

        # CSV download
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







