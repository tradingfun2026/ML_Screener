import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math, random

# ========================= SETTINGS =========================
THREADS=20
AUTO_REFRESH_MS=120_000
HISTORY_LOOKBACK_DAYS=10
INTRADAY_INTERVAL="2m"
INTRADAY_RANGE="1d"

DEFAULT_MAX_PRICE=5.0
DEFAULT_MIN_VOLUME=100_000
DEFAULT_MIN_BREAKOUT=0.0

st_autorefresh(interval=AUTO_REFRESH_MS,key="refresh_v9")

# ========================= PAGE =========================
st.set_page_config(page_title="V9 – 10-Day Momentum Screener",layout="wide")
st.title("🚀 V9 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized)")
st.caption("EMA10 • RSI7 • 3D/10D trend • RVOL10D • VWAP Flow • Alerts • Watchlist mode")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")
    watchlist_text=st.text_area("Watchlist tickers:",value="",height=80)
    watchlist_active=bool(watchlist_text.strip())

    ignore_filters_when_watchlist=st.checkbox(
        "Ignore filters when watchlist is populated",
        value=True
    )

    max_universe=st.slider("Max symbols to scan when no watchlist",
                           50,600,2000,50)

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode=st.radio("Universe Construction",[
        "Classic (Alphabetical Slice)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ])

    volume_rank_pool=st.slider("Volume-Ranked Scan Size (V9)",
                               100,2000,600,100)

    enable_enrichment=st.checkbox("Include float/short + news",value=False)

    st.markdown("---"); st.header("Filters")

    max_price=st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)

    # 🔥 updated to allow **0**
    min_volume=st.number_input("Min Daily Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)

    min_breakout=st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move=st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain=st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only=st.checkbox("Short-Squeeze Only")
    catalyst_only=st.checkbox("Must Have News/Earnings")
    vwap_only=st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---"); st.subheader("Order Flow Filter")
    enable_ofb_filter=st.checkbox("Use Min OFB Filter",value=False)
    min_ofb=st.slider("Min Order Flow Bias",0.00,1.00,0.50,0.01)

    st.markdown("---"); st.subheader("🔊 Audio Alerts")
    enable_alerts=st.checkbox("Enable Audio + Banner",value=False)
    ALERT_SCORE_THRESHOLD=st.slider("Alert when Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD=st.slider("Alert when Premarket % ≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD=st.slider("Alert when VWAP Dist % ≥",1,50,2,1)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — fresh scan will run.")

# ========================= LOAD SYMBOLS (FIXED) =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                       sep="|",skipfooter=1,engine="python")
    other=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                      sep="|",skipfooter=1,engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other=other.rename(columns={"ACT Symbol":"Symbol"})
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")

    # 🔥 FIX — ensures ETF column exists safely
    nasdaq=nasdaq.reindex(columns=["Symbol","ETF","Exchange"],fill_value=None)
    other =other.reindex(columns=["Symbol","ETF","Exchange"],fill_value=None)

    df=pd.concat([nasdaq,other]).dropna(subset=["Symbol"])
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ========================= UNIVERSE BUILDER =========================
def build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool):

    if watchlist_text.strip():
        raw=watchlist_text.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in raw]

    syms=load_symbols()

    if universe_mode=="Randomized Slice":
        random.shuffle(syms); return syms[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:volume_rank_pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty: ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except: continue
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_universe] if ranked else syms[:max_universe]

    return syms[:max_universe]

# ========================= SCAN ENGINE =========================
def scan_one(sym):
    try:
        t=yf.Ticker(sym["Symbol"])
        hist=t.history(period=f"{HISTORY_LOOKBACK_DAYS}d")
        if hist.empty: return None

        price=float(hist["Close"].iloc[-1])
        vol=float(hist["Volume"].iloc[-1])

        if not(ignore_filters_when_watchlist and watchlist_active):
            if price>max_price or vol<min_volume: return None

        close=hist["Close"]; delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rsi7=float((100-(100/(1+gain/loss))).iloc[-1])

        # 10D metrics
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>1 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>3 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # Live intraday
        intra=t.history(period="1d",interval="2m",prepost=True)
        vol_today=float(intra["Volume"].sum()) if not intra.empty else vol
        typical=(intra["High"]+intra["Low"]+intra["Close"])/3 if len(intra)>0 else None
        vwap_dist=None
        if typical is not None:
            twv=(typical*intra["Volume"]).sum(); tv=intra["Volume"].sum()
            if tv>0: vwap_dist=(price-(twv/tv))/(twv/tv)*100

        # Scoring
        rv10=vol_today/hist["Volume"].mean()
        score=max(0,yday or 0)*.8 + max(0,m3 or 0)*1.2 + max(0,m10 or 0)*.6 + rv10*2
        score=round(score,2)

        return{"Symbol":sym["Symbol"],"Price":price,"Volume":vol_today,
               "Score":score,"PM%":yday,"3D%":m3,"10D%":m10,
               "RSI7":round(rsi7,2),"RVOL_10D":round(rv10,2),
               "VWAP%":None if vwap_dist is None else round(vwap_dist,2)}
    except:return None

# ========================= MAIN SCAN =========================
@st.cache_data(ttl=8)
def run_scan():
    uni=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,u) for u in uni]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)

with st.spinner("Scanning…"):
    df=run_scan()

if df.empty: st.error("No results.")
else:
    if not(ignore_filters_when_watchlist and watchlist_active):
        df=df[df.Score>=min_breakout]

    st.subheader(f"{len(df)} results")
    for _,r in df.iterrows():
        st.write(r)



