###############################################
# FULL V10 — FIXED (FINAL SYMBOL LOADER PATCH)
###############################################

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math, random

# ========================================================
# CONFIG
# ========================================================
THREADS               = 20
AUTO_REFRESH_MS       = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================================================
st.set_page_config(page_title="V10 Hybrid Momentum Screener",layout="wide")
st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh_v10")
st.title("🚀 V10 — 10-Day Hybrid Momentum Screener (High Speed + Live Volume Mode)")

# ===================== Sidebar =========================
with st.sidebar:
    
    watchlist_text = st.text_area("Watchlist (any separator allowed)",value="",height=70)

    max_universe = st.slider("Max Universe (if no watchlist)",50,2000,2000,50)

    universe_mode = st.radio(
        "V10 Universe Mode",
        ["Classic (Alphabetical)","Randomized Slice","Live Volume Ranked (slower)"],
        index=0
    )
    volume_rank_pool = st.slider("Max pool for Volume-Rank mode",100,2000,600,100)

    enable_enrichment = st.checkbox("Include Float/Short/News (slower)",False)

    st.markdown("---")
    st.header("Filters")

    max_price  = st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)

    # Only approved UI change → min volume can be 0
    min_volume = st.number_input(
        "Min Daily Volume",
        0,               # ⬅ previously 10k
        50_000_000,
        DEFAULT_MIN_VOLUME,
        10_000
    )

    min_breakout = st.number_input("Min Score", -50.0,200.0,DEFAULT_MIN_BREAKOUT,1.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only   = st.checkbox("Short-Squeeze Only")
    catalyst_only  = st.checkbox("Must Have News")
    vwap_only      = st.checkbox("Above VWAP Only")

    st.subheader("Order Flow (Optional)")
    enable_ofb_filter = st.checkbox("Use Min Order-Flow Bias",False)
    min_ofb = st.slider("Min OFB (Buyer Strength Threshold)",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔔 Alerts")

    enable_alerts = st.checkbox("Enable Audio/Visual Alerts",False)
    ALERT_SCORE = st.slider("Alert if Score ≥",10,200,30,5)
    ALERT_PM    = st.slider("Alert if PM ≥",1,150,4,1)
    ALERT_VWAP  = st.slider("Alert if VWAP% ≥",1,50,2,1)

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared")

# ==================================================================================
# FIXED LOAD_SYMBOLS() — bulletproof, cannot KeyError now
# ==================================================================================
@st.cache_data(ttl=900)
def load_symbols():

    try:
        nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    except: nasdaq = pd.DataFrame()

    try:
        other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")
    except: other = pd.DataFrame()

    # Ensure Symbol column exists on both
    for df in (nasdaq,other):
        if "Symbol" not in df.columns:
            if "NASDAQ Symbol" in df.columns: df["Symbol"]=df["NASDAQ Symbol"]
            elif "ACT Symbol"    in df.columns: df["Symbol"]=df["ACT Symbol"]
            else: df["Symbol"]=None

        if "ETF" not in df.columns:
            df["ETF"]="N"
        if "Exchange" not in df.columns:
            df["Exchange"]="UNKNOWN"

    # 🔥 THIS IS THE FIX — slice only AFTER patches are applied
    keep_cols=["Symbol","ETF","Exchange"]
    nasdaq=nasdaq[keep_cols]
    other =other [keep_cols]

    df=pd.concat([nasdaq,other],ignore_index=True).dropna(subset=["Symbol"])
    df=df[df.Symbol.str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ==================================================================================
def build_universe(watchlist_text,max_universe,mode,volume_rank_pool):
    
    wl=watchlist_text.strip()
    if wl: 
        return [{"Symbol":t.upper(),"Exchange":"WATCH"} for t in wl.replace(","," ").replace("\n"," ").split()]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_universe]

    if mode=="Live Volume Ranked (slower)":
        ranked = []
        for s in syms[:volume_rank_pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty:
                    ranked.append({**s,"LiveVol":int(d["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_universe] if ranked else syms[:max_universe]

    return syms[:max_universe]

# ==================================================================================
# SCORING + AI commentary (unchanged)
# ==================================================================================
def short_score(pm,ydy,m3,m10,rsi,rvol,cat,sz,vwap,flow):
    s=0
    if pm: s+=pm*1.6
    if ydy:s+=ydy*0.8
    if m3: s+=m3*1.2
    if m10:s+=m10*0.6
    if rsi and rsi>55:s+=(rsi-55)*0.4
    if rvol and rvol>1.2:s+=(rvol-1.2)*2
    if vwap and vwap>0:s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if cat:s+=8
    if sz:s+=12
    return round(s,2)

def prob(score):
    try:return round((1/(1+math.exp(-score/20))*100),1)
    except:return None

def align(pm,m3,m10):
    b=sum([pm>0 if pm else 0, m3>0 if m3 else 0, m10>0 if m10 else 0])
    return["🔻 Not Aligned","🟡 Mixed","🟢 Bullish Lean","💚 Full Alignment"][b]

def ai(score,pm,rvol,flow,vwap,m10):
    out=[]
    if score>40:out.append("Momentum structure active")
    if pm>3:out.append("Strong premarket bid")
    if rvol>2:out.append("Liquidity expansion detected")
    if flow>0.65:out.append("Orderflow buyer-controlled")
    if vwap>0:out.append("Trading above VWAP")
    if m10>10:out.append("10D trend continuation")
    return " | ".join(out) if out else "Indecisive tape"

# ==================================================================================
# SCAN ENGINE
# ==================================================================================
def scan_one(sym,enrich,ofb,thresh):
    try:
        t=yf.Ticker(sym["Symbol"])
        d=t.history(period=f"{HISTORY_LOOKBACK_DAYS}d")
        if d.empty:return None

        price=float(d["Close"].iloc[-1])
        vol= int(d["Volume"].iloc[-1])
        if price>max_price or vol<min_volume: return None

        # 3/10 day
        ydy=(d["Close"].iloc[-1]-d["Close"].iloc[-2])/d["Close"].iloc[-2]*100 if len(d)>=2 else None
        m3 =(d["Close"].iloc[-1]-d["Close"].iloc[-4])/d["Close"].iloc[-4]*100 if len(d)>=4 else None
        m10=(d["Close"].iloc[-1]-d["Close"].iloc[0])/d["Close"].iloc[0]*100

        # RSI7
        delta=d["Close"].diff()
        rsi=100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/(-delta.clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])

        # RVOL
        rvol=vol/d["Volume"].mean()

        # Intraday
        intra=t.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None
        if not intra.empty:
            pm=(intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/intra["Close"].iloc[-2]*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            vwap=(tp*intra["Volume"]).sum()/intra["Volume"].sum()*100/intra["Close"].iloc[-1]
            sign=(intra["Close"]>intra["Open"])
            bv=(intra["Volume"]*sign).sum(); sv=(intra["Volume"]*~sign).sum()
            flow=bv/(bv+sv) if bv+sv>0 else None
            if ofb and (flow is None or flow<thresh):return None

        sz=low=float_flag=catalyst=False
        short=None;sector=industry="Unknown"

        if enrich:
            try:
                info=t.get_info()
                float_s=info.get("floatShares",None)
                short=info.get("shortPercentOfFloat",None)
                sector=info.get("sector","Unknown");industry=info.get("industry","Unknown")
                low=(float_s and float_s<20_000_000)
                sz=(short and short>0.15)
            except: pass
            try:
                n=t.get_news(); 
                if n: catalyst=(datetime.now(timezone.utc)-datetime.fromtimestamp(n[0]["providerPublishTime"],tz=timezone.utc)).days<=3
            except: pass

        score=short_score(pm,ydy,m3,m10,rsi,rvol,catalyst,sz,vwap,flow)

        return dict(
            Symbol=sym["Symbol"],Exchange=sym.get("Exchange"),
            Price=price,Volume=vol,Score=score,Prob_Rise=prob(score),
            PM=pm,YDay=ydy,Three=m3,Ten=m10,RSI=rsi,RVOL=rvol,VWAP=vwap,Flow=flow,
            Squeeze=sz,LowFloat=low,Short=short,News=catalyst,Sector=sector,Industry=industry,
            MTF=align(pm,m3,m10),AI=ai(score,pm,rvol,flow,vwap,m10),
            Spark=d["Close"]
        )
    except:return None

@st.cache_data(ttl=6)
def run_scan(wl,max_u,en,ofb,th,mode,pool):
    uni=build_universe(wl,max_u,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,u,en,ofb,th) for u in uni]):
            r=f.result(); 
            if r:out.append(r)
    return pd.DataFrame(out) if out else pd.DataFrame()

# ==================================================================================
with st.spinner("Running V10 Hybrid Scan…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty: st.error("No qualifying tickers — loosen filters or add watchlist.")
else:
    df=df[df.Score>=min_breakout]
    if min_pm_move: df=df[df.PM.fillna(-999)>=min_pm_move]
    if min_yday_gain:df=df[df.YDay.fillna(-999)>=min_yday_gain]
    if squeeze_only: df=df[df.Squeeze]
    if catalyst_only:df=df[df.News]
    if vwap_only:df=df[df.VWAP.fillna(-999)>0]
    df=df.sort_values("Score",ascending=False)

    st.subheader(f"Results: {len(df)} Matches")

    for _,r in df.iterrows():
        if enable_alerts and r.Score>=ALERT_SCORE:
            st.warning(f"🔔 {r.Symbol} Score {r.Score}")
        c1,c2,c3,c4=st.columns([2,3,3,3])

        c1.write(f"**{r.Symbol}** ({r.Exchange})")
        c1.write(f"💲 {r.Price} | 📊 Vol {r.Volume:,}")
        c1.write(f"🔥 Score {r.Score} | Prob {r.Prob_Rise}%")
        c1.write(r.MTF); c1.write(r.AI)

        c2.write(f"PM {r.PM}  YDay {r.YDay}")
        c2.write(f"3D {r.Three}  10D {r.Ten}")
        c2.write(f"RSI {r.RSI}  RVOL {round(r.RVOL,2)}x")

        c3.write(f"VWAP {round(r.VWAP,2)}%  Flow {round(r.Flow,2) if r.Flow else None}")
        if enable_enrichment:
            c3.write(f"Squeeze {r.Squeeze} | LowFloat {r.LowFloat}")
            c3.write(f"{r.Sector} / {r.Industry}")

        fig=go.Figure();fig.add_trace(go.Scatter(y=r.Spark.values,mode="lines"))
        fig.update_layout(height=70,margin=dict(l=0,r=0,t=0,b=0))
        c4.plotly_chart(fig,use_container_width=True)

        with c4.expander("📈 Open full chart"):
            fig2=go.Figure();fig2.add_trace(go.Scatter(y=r.Spark.values,mode="lines+markers"))
            fig2.update_layout(height=300,title=r.Symbol)
            st.plotly_chart(fig2,use_container_width=True)

        st.divider()



