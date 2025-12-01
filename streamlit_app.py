import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math, random

# ========================= SETTINGS =========================
THREADS = 20
AUTO_REFRESH_MS = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 5.0
DEFAULT_MIN_VOLUME = 0.0
DEFAULT_MIN_BREAKOUT = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v10")

# ========================= UI =========================
st.set_page_config(page_title="V10 Momentum Screener", layout="wide")

with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist:", "")

    max_universe = st.slider("Max symbols when NO watchlist",50,600,200,step=50)

    st.subheader("Scan Mode")
    universe_mode = st.radio("Universe Type",[
        "Classic","Randomized Slice","Live Volume Ranked (slower)"
    ])

    volume_rank_pool = st.slider("Volume Ranking Pool",100,2000,600,step=100)

    enable_enrichment = st.checkbox("Float/Short/News Enrichment (slower)",False)

    st.subheader("Filters")
    max_price = st.number_input("Max Price", 1.0, 1000.0, float(DEFAULT_MAX_PRICE),1.0)
    min_volume = st.number_input("Min Daily Volume",0.0,50_000_000.0,float(DEFAULT_MIN_VOLUME),10_000.0)
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain = st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("News Required")
    vwap_only = st.checkbox("Above VWAP Only")

    st.subheader("Order Flow Bias")
    enable_ofb_filter = st.checkbox("Enable Min OFB Filter")
    min_ofb = st.slider("Min Order Flow Bias (0–1)",0.0,1.0,0.50,0.01)

    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",True)
    ALERT_SCORE_THRESHOLD = st.slider("Score ≥",10,200,30)
    ALERT_PM_THRESHOLD = st.slider("PM% ≥",1,150,4)
    ALERT_VWAP_THRESHOLD = st.slider("VWAP% ≥",1,50,2)

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("OK — Cache Cleared.")

# ========================= FIXED — SYMBOL SOURCE =========================
@st.cache_data(ttl=900)
def load_symbols():

    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|",skipfooter=1,engine="python")
    other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                        sep="|",skipfooter=1,engine="python")

    # Column structure changed → force normalize
    nasdaq.rename(columns={"Symbol":"Symbol"},inplace=True)
    other.rename(columns={"ACT Symbol":"Symbol"},inplace=True)

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other.get("Exchange","OTHER")

    df=pd.concat([
        nasdaq[["Symbol","Exchange"]],
        other[["Symbol","Exchange"]]
    ],ignore_index=True)

    df=df[df.Symbol.str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ========================= UNIVERSE LOGIC =========================
def build_universe(wl,max_u,mode,pool):

    wl=wl.strip()
    if wl:
        items=set(w.upper() for w in wl.replace(","," ").split())
        return [{"Symbol":s,"Exchange":"WATCH"} for s in items]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_u]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m")
                if not d.empty:
                    ranked.append({**s,"V":float(d.Volume.iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("V",0),reverse=True)
        return ranked[:max_u] if ranked else syms[:max_u]

    return syms[:max_u]

# ========================= SCORING =========================
def score_calc(pm,m3,m10,rsi,rvol,vwap,ofb,cat,sq):
    s=0
    if pm: s+=pm*1.6
    if m3: s+=m3*1.2
    if m10: s+=m10*0.6
    if rsi>55: s+=(rsi-55)*0.4
    if rvol>1.2: s+=(rvol-1.2)*2
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if ofb: s+=(ofb-0.5)*22
    if cat: s+=8
    if sq:  s+=12
    return round(s,2)

def spark(series):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=series.values,mode="lines",line=dict(width=2)))
    fig.update_layout(height=60,width=160,margin=dict(l=0,r=0,t=0,b=0),
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ========================= SCAN ONE SYMBOL =========================
def scan(sym):
    try:
        t=yf.Ticker(sym["Symbol"])
        d=t.history(period="10d")
        if d.empty: return None

        price=float(d.Close.iloc[-1])
        vol=float(d.Volume.iloc[-1])
        if price>max_price or vol<min_volume: return None

        m3=(price-d.Close.iloc[-4])/d.Close.iloc[-4]*100 if len(d)>=4 else None
        m10=(price-d.Close.iloc[0])/d.Close.iloc[0]*100
        rsi=100-(100/(1+(d.Close.diff().clip(lower=0).rolling(7).mean()/(-d.Close.diff().clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])
        rvol=vol/d.Volume.mean()

        i=t.history(period="1d",interval="2m")
        if i.empty: return None

        last=float(i.Close.iloc[-1])
        prev=float(i.Close.iloc[-2])
        pm=(last-prev)/prev*100 if prev>0 else None

        tp=(i.High+i.Low+i.Close)/3
        v=(tp*i.Volume).sum()/i.Volume.sum() if i.Volume.sum()>0 else None
        vwap=(price-v)/v*100 if v else None

        sign=(i.Close>i.Open).astype(int)-(i.Close<i.Open).astype(int)
        buy=(i.Volume*(sign>0)).sum(); sell=(i.Volume*(sign<0)).sum()
        ofb=buy/(buy+sell) if buy+sell>0 else None
        if enable_ofb_filter and (ofb is None or ofb<min_ofb): return None

        score=score_calc(pm,m3,m10,rsi,rvol,vwap,ofb,False,False)
        prob=round((1/(1+math.exp(-score/20)))*100,2)

        return {
            "Symbol":sym["Symbol"],
            "Price":round(price,2),
            "Volume":int(vol),
            "Score":score,
            "Prob%":prob,
            "PM%":round(pm,2) if pm else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),
            "RSI7":round(rsi,2),
            "RVOL":round(rvol,2),
            "VWAP%":round(vwap,2) if vwap else None,
            "Flow":round(ofb,2) if ofb else None,
            "Spark":d.Close
        }

    except:
        return None

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
if df.empty: st.error("No results found."); st.stop()

df=df[df.Score>=min_breakout]
df=df.sort_values(["Score","PM%","Flow"],ascending=[False,False,False])

st.subheader(f"🔥 V10 Results — {len(df)}")

for _,r in df.iterrows():

    sym=r.Symbol

    if enable_alerts and sym not in st.session_state.alerted:
        if r.Score>=ALERT_SCORE_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} — Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} — Premarket {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} — VWAP {r['VWAP%']}%")

    c1,c2,c3=st.columns([2,2,3])
    c1.markdown(f"### {sym} — ${r.Price}")
    c1.write(f"Volume {r.Volume:,}")
    c1.write(f"🔥 Score {r.Score} | Prob {r['Prob%']}%")

    c2.write(f"PM {r['PM%']}% | 3D {r['3D%']}% | 10D {r['10D%']}%")
    c2.write(f"RSI {r.RSI7} | RVOL {r.RVOL}x")
    c2.write(f"VWAP {r['VWAP%']}% | Flow {r['Flow']}")

    c3.plotly_chart(spark(r.Spark),use_container_width=False)
    st.divider()

st.download_button("📥 Export CSV",df.to_csv(index=False),"V10_output.csv")

st.download_button("📥 Export CSV",df.to_csv(index=False),"V10_output.csv")


