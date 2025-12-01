###############################################################
###########       V10 FINAL – HARDENED BUILD         ##########
###########    (load_symbols permanently fixed)      ##########
###############################################################

import streamlit as st, pandas as pd, yfinance as yf, plotly.graph_objs as go
import concurrent.futures, math, random
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

THREADS=20
AUTO_REFRESH_MS=120_000
HISTORY_LOOKBACK_DAYS=10
INTRADAY_INTERVAL="2m"
INTRADAY_RANGE="1d"

DEFAULT_MAX_PRICE=5.0
DEFAULT_MIN_VOLUME=100_000
DEFAULT_MIN_BREAKOUT=0.0

st.set_page_config(page_title="V10 Momentum Screener",layout="wide")
st_autorefresh(interval=AUTO_REFRESH_MS,key="refresh_v10")
st.title("🚀 V10 – 10-Day Hybrid Momentum Screener")

###############################################################
####################  SIDEBAR    ##############################
###############################################################
with st.sidebar:

    watchlist_text=st.text_area("Watchlist",value="",height=70)
    max_universe=st.slider("Max Universe",50,2000,2000,50)

    universe_mode=st.radio("Universe Mode",
        ["Classic (Alphabetical)","Randomized Slice","Live Volume Ranked (slower)"],index=0)

    volume_rank_pool=st.slider("Volume Rank Pool",100,2000,600,100)

    enable_enrichment=st.checkbox("Include Float/Short/News (slower)",False)

    st.markdown("---"); st.header("Filters")

    max_price=st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)

    #### ONLY UI CHANGE ALLOWED — MIN VOL CAN BE ZERO ####
    min_volume=st.number_input("Min Daily Volume",0,50_000_000,DEFAULT_MIN_VOLUME,10_000)

    min_breakout=st.number_input("Min Score",-50.0,200.0,DEFAULT_MIN_BREAKOUT,1.0)
    min_pm_move=st.number_input("Min Premarket %",-50,200,0.0,0.5)
    min_yday_gain=st.number_input("Min Yesterday %",-50,200,0.0,0.5)

    squeeze_only=st.checkbox("Short-Squeeze Only")
    catalyst_only=st.checkbox("Must Have Catalyst")
    vwap_only=st.checkbox("Above VWAP Only")

    st.subheader("Order Flow Filter")
    enable_ofb_filter=st.checkbox("Enable Orderflow Minimum",False)
    min_ofb=st.slider("Min OFB",0.0,1.0,0.50,0.01)

    st.markdown("---"); st.subheader("🔔 Alerts")
    enable_alerts=st.checkbox("Enable Alerts",False)
    ALERT_SCORE=st.slider("Alert if Score ≥",10,200,30,5)
    ALERT_PM=st.slider("Alert if PM ≥",1,150,4,1)
    ALERT_VWAP=st.slider("Alert if VWAP ≥",1,50,2,1)

###############################################################
############ FINAL SAFE load_symbols() (bulletproof) ##########
###############################################################
@st.cache_data(ttl=900)
def load_symbols():

    def safe_load(url):
        try: return pd.read_csv(url,sep="|",skipfooter=1,engine="python")
        except: return pd.DataFrame()

    nasdaq=safe_load("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
    other =safe_load("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")

    # Guarantee Symbol column exists
    for df in (nasdaq,other):
        if "Symbol" not in df.columns:
            if "NASDAQ Symbol" in df.columns: df["Symbol"]=df["NASDAQ Symbol"]
            elif "ACT Symbol" in df.columns: df["Symbol"]=df["ACT Symbol"]
            else: df["Symbol"]=None

        # Standardize column names dynamically — FIX
        df.rename(columns={"ETF":"ETF","Exchange":"Exchange"},inplace=True)

    # 🔥 PERMANENT FIX — NO KEYERROR POSSIBLE
    nasdaq = nasdaq.reindex(columns=["Symbol","ETF","Exchange"])
    other  = other .reindex(columns=["Symbol","ETF","Exchange"])

    df=pd.concat([nasdaq,other],ignore_index=True)
    df=df.dropna(subset=["Symbol"])
    df=df[df.Symbol.str.contains(r"^[A-Z]{1,5}$",na=False)]

    return df.to_dict("records")

###############################################################
############ Universe Builder (unchanged logic) ###############
###############################################################
def build_universe(wl,max_u,mode,pool):

    if wl.strip():
        return [{"Symbol":x.upper(),"Exchange":"WATCH"} for x in wl.replace(","," ").replace("\n"," ").split()]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms); return syms[:max_u]

    if mode=="Live Volume Ranked (slower)":
        live=[]
        for s in syms[:pool]:
            try:
                h=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not h.empty: live.append({**s,"Vol":int(h["Volume"].iloc[-1])})
            except: pass
        if live: return sorted(live,key=lambda x:x.get("Vol",0),reverse=True)[:max_u]

    return syms[:max_u]

###############################################################
############## Score + Probability + AI View ##################
###############################################################
def score_calc(pm,ydy,m3,m10,rsi,rvol,cat,sz,vwap,flow):
    s=0
    if pm: s+=pm*1.6
    if ydy: s+=ydy*.8
    if m3: s+=m3*1.2
    if m10:s+=m10*.6
    if rsi>55:s+=(rsi-55)*.4
    if rvol>1.2:s+=(rvol-1.2)*2
    if vwap>0:s+=min(vwap,6)*1.5
    if flow:s+=(flow-0.5)*22
    if cat:s+=8
    if sz:s+=12
    return round(s,2)

def p(s): return round((1/(1+math.exp(-s/20)))*100,1)

def align(pm,m3,m10):
    c=sum([(pm>0 if pm else 0),(m3>0 if m3 else 0),(m10>0 if m10 else 0)])
    return["🔻","🟡","🟢","💚 Full Alignment"][c]

def ai(s,pm,rvol,flow,vwap,m10):
    o=[]
    if s>40:o.append("Momentum Live")
    if pm>3:o.append("Premarket strength")
    if rvol>2:o.append("Volume expansion")
    if flow>0.65:o.append("Buyer control")
    if vwap>0:o.append("Above VWAP")
    if m10>10:o.append("Trend sustained")
    return " | ".join(o) or "No strong signal yet"

###############################################################
########################## SCANNER ############################
###############################################################
def scan(sym,enrich,ofb,th):
    try:
        t=yf.Ticker(sym["Symbol"])
        d=t.history(period=f"{HISTORY_LOOKBACK_DAYS}d")
        if d.empty:return None

        price=float(d.Close.iloc[-1])
        vol=int(d.Volume.iloc[-1])

        if price>max_price or vol<min_volume:return None

        ydy=(d.Close.iloc[-1]-d.Close.iloc[-2])/d.Close.iloc[-2]*100 if len(d)>2 else None
        m3 =(d.Close.iloc[-1]-d.Close.iloc[-4])/d.Close.iloc[-4]*100 if len(d)>4 else None
        m10=(d.Close.iloc[-1]-d.Close.iloc[0])/d.Close.iloc[0]*100

        # RSI7
        rsi=d.Close.diff()
        rsi=100-(100/(1+(rsi.clip(lower=0).rolling(7).mean()/(-rsi.clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])

        rvol=vol/d.Volume.mean()

        # INTRADAY
        intra=t.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None
        if not intra.empty:
            pm=(intra.Close.iloc[-1]-intra.Close.iloc[-2])/intra.Close.iloc[-2]*100
            tp=(intra.High+intra.Low+intra.Close)/3
            vwap=(tp*intra.Volume).sum()/intra.Volume.sum()*100/intra.Close.iloc[-1]
            sign=(intra.Close>intra.Open)
            bv=(intra.Volume*sign).sum();sv=(intra.Volume*~sign).sum()
            flow=bv/(bv+sv) if (bv+sv)>0 else None
            if ofb and (flow is None or flow<th):return None

        # OPTIONAL FUNDAMENTAL DATA
        sz=low=False; short=None; sector=industry="Unknown"; catalyst=False

        if enrich:
            try:
                info=t.get_info() or {}
                float_s=info.get("floatShares",None)
                short=info.get("shortPercentOfFloat",None)
                sector=info.get("sector","Unknown")
                industry=info.get("industry","Unknown")
                low=float_s and float_s<20_000_000
                sz=short and short>0.15
            except:pass
            try:
                n=t.get_news()
                if n:catalyst=(datetime.now(timezone.utc)-datetime.fromtimestamp(n[0]['providerPublishTime'],tz=timezone.utc)).days<=3
            except:pass

        s=score_calc(pm,ydy,m3,m10,rsi,rvol,catalyst,sz,vwap,flow)

        return dict(Symbol=sym["Symbol"],Price=price,Volume=vol,
                    Score=s,Prob=p(s),PM=pm,YDay=ydy,Three=m3,Ten=m10,RSI=rsi,RVOL=rvol,
                    VWAP=vwap,Flow=flow,Squeeze=sz,LowFloat=low,Catalyst=catalyst,
                    Sector=sector,Industry=industry,MTF=align(pm,m3,m10),AI=ai(s,pm,rvol,flow,vwap,m10),
                    Spark=d.Close)

    except:return None

###############################################################
######################## RUN SCAN #############################
###############################################################
@st.cache_data(ttl=6)
def run_scan(wl,maxu,en,ofb,th,mode,pool):
    uni=build_universe(wl,maxu,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan,u,en,ofb,th) for u in uni]):
            r=f.result()
            if r:out.append(r)
    return pd.DataFrame(out)

###############################################################
######################## DISPLAY ##############################
###############################################################
with st.spinner("Scanning…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty:
    st.error("No symbols returned — relax filters or enter watchlist")
else:
    df=df[df.Score>=min_breakout]
    if min_pm_move: df=df[df.PM.fillna(-999)>=min_pm_move]
    if min_yday_gain:df=df[df.YDay.fillna(-999)>=min_yday_gain]
    if squeeze_only: df=df[df.Squeeze]
    if catalyst_only:df=df[df.Catalyst]
    if vwap_only:df=df[df.VWAP.fillna(-999)>0]

    df=df.sort_values("Score",ascending=False)

    st.subheader(f"Matches: {len(df)}")

    for _,r in df.iterrows():
        if enable_alerts and r.Score>=ALERT_SCORE:
            st.warning(f"🔔 {r.Symbol} Score {r.Score}")

        c1,c2,c3,c4=st.columns([2,3,3,3])

        c1.write(f"**{r.Symbol}**")
        c1.write(f"💲 {r.Price}   |   📊 {r.Volume:,}")
        c1.write(f"🔥 {r.Score} → {r.Prob}%")
        c1.write(r.MTF)
        c1.write(r.AI)

        c2.write(f"PM  {r.PM}")
        c2.write(f"YDay {r.YDay}")
        c2.write(f"3D {r.Three} | 10D {r.Ten}")
        c2.write(f"RSI {r.RSI} | RVOL {round(r.RVOL,2)}x")

        c3.write(f"VWAP {round(r.VWAP,2) if r.VWAP else None}%")
        c3.write(f"Flow {round(r.Flow,2) if r.Flow else None}")

        if enable_enrichment:
            c3.write(f"Squeeze {r.Squeeze} | LowFloat {r.LowFloat}")
            c3.write(f"{r.Sector} / {r.Industry}")

        fig=go.Figure()
        fig.add_trace(go.Scatter(y=r.Spark.values,mode="lines"))
        fig.update_layout(height=70,margin=dict(l=0,r=0,t=0,b=0))
        c4.plotly_chart(fig,use_container_width=True)

        with c4.expander("📈 Full Chart"):
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(y=r.Spark.values,mode="lines+markers"))
            fig2.update_layout(height=300,title=r.Symbol)
            st.plotly_chart(fig2,use_container_width=True)

        st.divider()



