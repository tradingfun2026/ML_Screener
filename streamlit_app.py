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
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 5.0
DEFAULT_MIN_VOLUME = 0.0            # << now allows 0 min volume
DEFAULT_MIN_BREAKOUT = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v10")

# ========================= PAGE =========================
st.set_page_config(page_title="V10 — Momentum AI Screener", layout="wide")
st.title("🚀 V10 — 10-Day Momentum Breakout Screener")
st.caption("All UI preserved • Volume fixed • Fast V10 scan engine • Watchlist, VWAP, PM, Flow, RVOL, AI commentary")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers:", "")

    max_universe = st.slider("Max symbols to scan when no watchlist",50,600,200,step=50)

    st.subheader("V10 Universe Mode")
    universe_mode = st.radio("Universe type",["Classic","Randomized Slice","Live Volume Ranked (slower)"],index=0)
    volume_rank_pool = st.slider("Symbols to evaluate in Live Volume mode",100,2000,600,step=100)

    enable_enrichment = st.checkbox("Float/Short/News (slower)",False)

    st.subheader("Filters")
    max_price = st.number_input("Max Price", 1.0, 1000.0, float(DEFAULT_MAX_PRICE), 1.0)   # float-corrected
    min_volume = st.number_input("Min Daily Volume", 0.0, 50_000_000.0, float(DEFAULT_MIN_VOLUME), 10_000.0)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, float(DEFAULT_MIN_BREAKOUT), 1.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("News Required")
    vwap_only = st.checkbox("Above VWAP Only")

    st.subheader("Order Flow")
    enable_ofb_filter = st.checkbox("Enable Flow Bias Filter")
    min_ofb = st.slider("Minimum Flow Bias (0–1)",0.0,1.0,0.50,0.01)

    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",True)
    ALERT_SCORE_THRESHOLD = st.slider("Trigger Score ≥",10,200,30)
    ALERT_PM_THRESHOLD = st.slider("Trigger PM% ≥",1,150,4)
    ALERT_VWAP_THRESHOLD = st.slider("Trigger VWAP% ≥",1,50,2)

    if st.button("🔄 Reset Cache"):
        st.cache_data.clear()
        st.success("Cache cleared — reload to rescan")

# ========================= LOAD SYMBOLS — FIXED =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")

    nasdaq.rename(columns={"Symbol":"Symbol"},inplace=True)
    other.rename(columns={"ACT Symbol":"Symbol"},inplace=True)

    df=pd.concat([nasdaq[["Symbol","Exchange"]],other[["Symbol","Exchange"]]],ignore_index=True).dropna()
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ========================= UNIVERSE BUILDER V10 =========================
def build_universe(wl,max_u,mode,pool):
    wl=wl.strip()

    # If watchlist provided → use EXACTLY watchlist (filters still apply)
    if wl:
        raw=wl.replace(","," ").replace("\n"," ").split()
        tick=set(x.upper().strip() for x in raw if len(x)>0)
        return [{"Symbol":t,"Exc":"WL"} for t in tick]

    syms=load_symbols()

    if mode=="Randomized Slice":
        r=syms[:] ; random.shuffle(r)
        return r[:max_u]

    if mode=="Live Volume Ranked (slower)":
        base=syms[:pool]
        ranked=[]
        for s in base:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m")
                if not d.empty:
                    ranked.append({**s,"vol":float(d["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("vol",0),reverse=True)
        return ranked[:max_u] if ranked else syms[:max_u]

    return syms[:max_u]

# ========================= V10 SCORING =========================
def v10_score(pm,y3,y10,rsi,rvol,vwap,flow,catalyst,sq):
    s=0
    if pm  is not None: s+=pm*1.6
    if y3  is not None: s+=y3*1.2
    if y10 is not None: s+=y10*0.6
    if rsi and rsi>55:  s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if catalyst: s+=8
    if sq: s+=12
    return round(s,2)

def sparkline(s):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=s.values,mode="lines",line=dict(width=2)))
    fig.update_layout(height=65,width=150,margin=dict(l=0,r=0,t=0,b=0),xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ========================= SCAN =========================
def scan(sym):
    try:
        t=yf.Ticker(sym["Symbol"])
        h=t.history(period="10d")
        if h.empty: return None

        price=float(h["Close"].iloc[-1])
        vol=float(h["Volume"].iloc[-1])
        if price>max_price or vol<min_volume: return None   # << includes realtime volume

        y3=(price-h["Close"].iloc[-4])/h["Close"].iloc[-4]*100 if len(h)>=4 else None
        y10=(price-h["Close"].iloc[0])/h["Close"].iloc[0]*100

        rsi=100-(100/(1+(h["Close"].diff().clip(lower=0).rolling(7).mean()/(-h["Close"].diff().clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])

        rvol=vol/h["Volume"].mean()

        # Intraday PM, VWAP
        i=t.history(period="1d",interval="2m")
        if i.empty: return None

        last=float(i["Close"].iloc[-1])
        prev=float(i["Close"].iloc[-2])
        pm=(last-prev)/prev*100 if prev>0 else None

        tp=(i["High"]+i["Low"]+i["Close"])/3
        vwap=(price-(tp*i["Volume"]).sum()/i["Volume"].sum())/((tp*i["Volume"]).sum()/i["Volume"].sum())*100 if i["Volume"].sum()>0 else None

        # Order Flow Bias
        sign=(i["Close"]>i["Open"]).astype(int)-(i["Close"]<i["Open"]).astype(int)
        buy=(i["Volume"]*(sign>0)).sum(); sell=(i["Volume"]*(sign<0)).sum()
        flow=buy/(buy+sell) if (buy+sell)>0 else None

        if enable_ofb_filter and (flow is None or flow<min_ofb): return None

        score=v10_score(pm,y3,y10,rsi,rvol,vwap,flow,False,False)
        prob=round((1/(1+math.exp(-score/20)))*100,2)

        return {
            "Symbol":sym["Symbol"],
            "Price":round(price,2),
            "Volume":int(vol),            # << now realtime visible
            "Score":score,
            "Prob%":prob,
            "PM%": round(pm,2) if pm else None,
            "3D%": round(y3,2) if y3 else None,
            "10D%":round(y10,2),
            "RSI7":round(rsi,2),
            "RVOL":round(rvol,2),
            "VWAP%":round(vwap,2) if vwap else None,
            "Flow":round(flow,2) if flow else None,
            "Spark":h["Close"]
        }
    except: return None

# ========================= RUN =========================
if "alerted" not in st.session_state: st.session_state.alerted=set()

with st.spinner("Scanning V10…"):
    uni=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan,u) for u in uni]):
            r=f.result()
            if r: out.append(r)

df=pd.DataFrame(out)
if df.empty: st.error("No results found. Adjust filters."); st.stop()

df=df[df.Score>=min_breakout]
df=df.sort_values(["Score","PM%","Flow"],ascending=[False,False,False])

st.subheader(f"🔥 V10 Results — {len(df)}")

# ========================= DISPLAY =========================
for _,r in df.iterrows():

    sym=r["Symbol"]

    if enable_alerts and sym not in st.session_state.alerted:
        if r["Score"]>=ALERT_SCORE_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} — Score {r['Score']}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} Premarket {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} VWAP {r['VWAP%']}%")

    c1,c2,c3=st.columns([2,2,3])

    c1.markdown(f"### {sym} — ${r['Price']}")
    c1.write(f"Volume **{r['Volume']:,}**")
    c1.write(f"Score {r['Score']}  | Prob {r['Prob%']}%")

    c2.write(f"PM {r['PM%']}%  |  3D {r['3D%']}%")
    c2.write(f"10D {r['10D%']}% | RSI7 {r['RSI7']}")
    c2.write(f"VWAP {r['VWAP%']}% | Flow {r['Flow']}")

    c3.plotly_chart(sparkline(r["Spark"]),use_container_width=False)
    st.divider()

st.download_button("📥 Export CSV",df.to_csv(index=False),"V10_output.csv")


