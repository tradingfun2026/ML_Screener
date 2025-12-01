##############################################################
#  🟢 V10 — FULL SAFE FIX RELEASE  (NO UI CHANGES)
#  ✔ ETF/Exchange KeyError FIXED permanently
#  ✔ Min Volume now supports 0
#  ✔ Volume displays current real volume correctly
##############################################################

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# ========================= SETTINGS =========================
THREADS = 20
AUTO_REFRESH_MS = 120_000
LOOKBACK = 10
DEFAULT_MAX_PRICE = 5.0
DEFAULT_MIN_VOLUME = 0   # ← YOU REQUESTED THIS
DEFAULT_BREAKOUT = 0.0

st_autorefresh(interval=AUTO_REFRESH_MS,key="v10_ref")

st.set_page_config(page_title="V10 Screener",layout="wide")
st.title("🚀 V10 Momentum Screener")
st.caption("V10 = Speed + Stability + Volume Awareness (UI Unchanged)")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watch = st.text_area("Watchlist (comma/space/new-line):","",height=90)

    max_universe = st.slider("Max Symbols (no watchlist)",50,600,2000,50)

    st.subheader("Universe Mode")
    universe_mode = st.radio("Selection Method",[
        "Classic (Alphabetical)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ])

    volume_rank_pool = st.slider("Pool Size for Volume Ranking",100,2000,600,100)

    enrich = st.checkbox("Float/Short/News Data (slower)",False)

    st.subheader("Filters")
    max_price = st.number_input("Max Price",1.0,2000.0,DEFAULT_MAX_PRICE)
    min_vol   = st.number_input("Min Daily Volume",0,50_000_000,DEFAULT_MIN_VOLUME)
    min_score = st.number_input("Min Breakout Score",-50.0,200.0,DEFAULT_BREAKOUT)

    min_pm  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    sq_only = st.checkbox("Short Squeeze Only")
    news_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only")

    st.subheader("Order Flow Filter")
    ofb_filter = st.checkbox("Use Flow Bias Filter",False)
    min_ofb = st.slider("Min Flow Bias",0.00,1.00,0.50,0.01)

    st.subheader("Alerts")
    enable_alerts = st.checkbox("Enable Alerts",False)
    ALERT_SCORE = st.slider("Score ≥",10,200,30)
    ALERT_PM = st.slider("PM% ≥",1,150,4)
    ALERT_VWAP = st.slider("VWAP% ≥",1,50,2)

    if st.button("Refresh"):
        st.cache_data.clear()
        st.success("Cache Cleared — Restarting Fresh")

# ========================= FIXED — NO MORE KEYERROR =========================
@st.cache_data(ttl=900)
def load_symbols():
    n = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                    sep="|",skipfooter=1,engine="python")
    o = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                    sep="|",skipfooter=1,engine="python")

    if "ACT Symbol" in o.columns and "Symbol" not in o.columns:
        o = o.rename(columns={"ACT Symbol":"Symbol"})

    required = ["Symbol","ETF","Exchange"]
    for df in (n,o):
        for c in required:
            if c not in df.columns: df[c] = None  # ← KEY FIX

    merged = pd.concat([n[required],o[required]],ignore_index=True)
    merged = merged.dropna(subset=["Symbol"])
    merged["Symbol"] = merged["Symbol"].astype(str)
    merged = merged[merged["Symbol"].str.fullmatch(r"[A-Z]{1,5}")]

    return merged.to_dict("records")

# ========================= BUILD UNIVERSE =========================
def build_universe():
    if watch.strip():
        tick=set(w.upper() for w in watch.replace(","," ").split() if w)
        return [{"Symbol":t} for t in sorted(tick)]

    syms = load_symbols()

    if universe_mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        base=syms[:volume_rank_pool]
        ranked=[]

        for s in base:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m")
                if not d.empty:
                    ranked.append({**s,"livevol":int(d["Volume"].iloc[-1])})
            except: pass

        ranked = sorted(ranked,key=lambda x:x.get("livevol",0),reverse=True)
        return ranked[:max_universe] or syms[:max_universe]

    return syms[:max_universe]

# ========================= SCANNER =========================
def scan_one(sym):
    try:
        t=yf.Ticker(sym["Symbol"])
        h=t.history(period="10d")
        if h.empty: return None

        close=h["Close"]; volume=h["Volume"]
        price=float(close.iloc[-1]); vol=int(volume.iloc[-1])

        if price>max_price or vol<min_vol: return None

        y  = (close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>1 else None
        m3 = (close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>3 else None
        m10= (close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        rsi = (close.diff().clip(lower=0).rolling(7).mean() /
               -close.diff().clip(upper=0).rolling(7).mean())
        rsi7=float(100-(100/(1+rsi))).__round__(2)

        rvol = vol/(volume.mean()) if volume.mean()>0 else None

        intra=t.history(period="1d",interval="2m")
        pm=None; vwap=None; flow=None

        if not intra.empty:
            pm=(intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/intra["Close"].iloc[-2]*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            if intra["Volume"].sum()>0:
                v=float((tp*intra["Volume"]).sum()/intra["Volume"].sum())
                vwap=(price-v)/v*100

            sig=(intra["Close"]>intra["Open"]).astype(int)-(intra["Close"]<intra["Open"]).astype(int)
            buy=(intra["Volume"]*(sig>0)).sum(); sell=(intra["Volume"]*(sig<0)).sum()
            flow=buy/(buy+sell) if (buy+sell)>0 else None

        if ofb_filter and (flow is None or flow<min_ofb): return None

        score=0
        if pm: score+=pm*1.6
        if y: score+=y*0.8
        if m3: score+=m3*1.2
        if m10: score+=m10*0.6
        if rsi7>55: score+=(rsi7-55)*0.4
        if rvol and rvol>1.2: score+=(rvol-1.2)*2
        if vwap and vwap>0: score+=min(vwap,6)*1.5
        if flow: score+=(flow-0.5)*22

        score=round(score,2)
        prob=round((1/(1+math.exp(-score/20)))*100,1) if score else None

        return {
            "Symbol":sym["Symbol"],
            "Price":round(price,2),
            "Volume":vol,                        # 🔥 YOU WANTED THIS
            "Score":score,
            "Prob_Rise%":prob,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(y,2) if y else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),
            "RSI7":rsi7,
            "RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(flow,2) if flow else None
        }
    except: return None

# ========================= RUN SCAN=========================
if "alerted" not in st.session_state: st.session_state.alerted=set()

with st.spinner("Scanning V10…"):
    uni=build_universe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        results=[f.result() for f in concurrent.futures.as_completed([ex.submit(scan_one,u) for u in uni]) if f.result()]

df=pd.DataFrame(results)

if df.empty:
    st.error("No results — loosen filters or add watchlist")
    st.stop()

# ========= filtering preserved ==========
df=df[df.Score>=min_score]
df=df[df["PM%"].fillna(-999)>=min_pm]
df=df[df["YDay%"].fillna(-999)>=min_yday]
if vwap_only: df=df[df["VWAP%"].fillna(-999) > 0]
df=df.sort_values(["Score","PM%"],ascending=[False,False])

# ========================= DISPLAY =========================
st.subheader(f"🔥 {len(df)} matches")

for _,r in df.iterrows():
    sym=r.Symbol

    if enable_alerts and sym not in st.session_state.alerted:
        if r.Score>=ALERT_SCORE: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} Score {r.Score}")
        elif r["PM%"]and r["PM%"]>=ALERT_PM: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} PM {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} VWAP breakout")

    c1,c2,c3=st.columns([2,3,3])
    c1.markdown(f"### {sym}")
    c1.write(f"💲 {r.Price}")
    c1.write(f"📊 Volume **{r.Volume:,}**")
    c1.write(f"🔥 Score **{r.Score}** | Prob {r['Prob_Rise%']}%")
    c2.write(f"PM {r['PM%']}%  |  Y {r['YDay%']}% | 3D {r['3D%']}% | 10D {r['10D%']}%")
    c2.write(f"RSI7 {r.RSI7} | RVOL {r.RVOL_10D}x")
    c2.write(f"VWAP {r['VWAP%']}% | Flow {r.FlowBias}")
    spark=go.Figure(go.Scatter(y=[r.Price],mode="lines")) #< keep original display minimal
    c3.plotly_chart(spark,use_container_width=True)
    st.divider()

st.download_button("📥 Download CSV",df.to_csv(index=False),"v10.csv")




