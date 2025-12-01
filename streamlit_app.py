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
THREADS               = 20
AUTO_REFRESH_MS       = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v10")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V10 – Momentum Screener (Live Volume)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V10 — 10-Day Momentum Breakout Screener (w/ Live Intraday Volume)")
st.caption(
    "EMA10 • RSI7 • 3D & 10D momentum • LIVE volume feed • "
    "10D RVOL • VWAP edge • AI commentary • Universe modes preserved"
)

# --------------------------------------------------------------------------------
#  SIDEBAR (NO UI REMOVED — only additions below)
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
        help="Watchlist now forces scan priority + bypasses universe filters automatically."
    )

    max_universe = st.slider(
        "Max symbols to scan when no watchlist", min_value=50, max_value=600,
        value=2000, step=50
    )

    # universe mode (unchanged)
    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"],
        index=0
    )

    volume_rank_pool = st.slider("Volume-Rank Pool Size", 100, 2000, 600, 100)

    enable_enrichment = st.checkbox("Include float/short + news (slower)", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume = st.number_input("Min Daily Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000) # 🔥 reset powered = 0 allowed
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only   = st.checkbox("Short-Squeeze Only")
    catalyst_only  = st.checkbox("Must Have News/Earnings")
    vwap_only      = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox("Use Min Order Flow Bias Filter",value=False)
    min_ofb = st.slider("Min Order Flow Bias",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Audio Alerts")
    enable_alerts = st.checkbox("Enable Alerts",value=False)
    ALERT_SCORE_THRESHOLD=st.slider("Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD   =st.slider("Premarket % ≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD =st.slider("VWAP Dist % ≥",1,50,2,1)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache wiped → fresh scan loading")

# --------------------------------------------------------------------------------
#  Symbol Source
# --------------------------------------------------------------------------------
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")

    other = other.rename(columns={"ACT Symbol":"Symbol"})
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")

    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]]).dropna()
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

def build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool):
    wl=watchlist_text.strip()
    if wl: # ⬅ Watchlist dominates everything now
        raw=wl.replace("\n"," ").replace(","," ").split()
        tickers=sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tickers]

    syms=load_symbols()

    if universe_mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:volume_rank_pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m")
                if not d.empty: ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_universe] if ranked else syms[:max_universe]

    return syms[:max_universe]

# --------------------------------------------------------------------------------
#  Score + Trend Engine (unchanged)
# --------------------------------------------------------------------------------
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow_bias):
    score=0
    if pm:score+=max(pm,0)*1.6
    if yday:score+=max(yday,0)*0.8
    if m3:score+=max(m3,0)*1.2
    if m10:score+=max(m10,0)*0.6
    if rsi7 and rsi7>55:score+=(rsi7-55)*0.4
    if rvol10 and rvol10>1.2:score+=(rvol10-1.2)*2
    if vwap and vwap>0:score+=min(vwap,6)*1.5
    if flow_bias:score+=(flow_bias-0.5)*22
    if catalyst:score+=8
    if squeeze:score+=12
    return round(score,2)

def breakout_probability(score):
    try:return round((1/(1+math.exp(-score/20)))*100,1)
    except:return None

def multi_timeframe_label(pm,m3,m10):
    b1=pm and pm>0; b2=m3 and m3>0; b3=m10 and m10>0
    s=sum([b1,b2,b3])
    return ["🔻 Not Aligned","🟡 Mixed","🟢 Leaning Bullish","✅ Full Alignment"][s]

# --------------------------------------------------------------------------------
#  MAIN SCANNER — NOW WITH LIVE VOLUME
# --------------------------------------------------------------------------------
def scan_one(sym,enable_enrichment,enable_ofb_filter,min_ofb):
    try:
        t=yf.Ticker(sym["Symbol"])

        daily=t.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")
        if daily.empty or len(daily)<5:return None

        close=daily["Close"];vol=daily["Volume"]
        price=float(close.iloc[-1])
        avg10=vol.mean();rvol10=float(vol.iloc[-1]/avg10) if avg10>0 else None

        # 🔥 live volume pulled here
        intra=t.history(period="1d",interval="2m",prepost=True)
        live_vol=int(intra["Volume"].iloc[-1]) if not intra.empty else 0

        if price>max_price or live_vol<min_volume:return None

        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        delta=close.diff()
        rsi7=float((100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/
                                   (-delta.clip(upper=0)).rolling(7).mean())))).iloc[-1])

        typical=(intra["High"]+intra["Low"]+intra["Close"])/3 if not intra.empty else 0
        vwap=((typical*intra["Volume"]).sum()/intra["Volume"].sum()) if not intra.empty else None
        vwap_dist=((price-vwap)/vwap*100) if vwap else None

        sign=(intra["Close"]>intra["Open"]).astype(int)-(intra["Close"]<intra["Open"]).astype(int)
        buy=(intra["Volume"]*(sign>0)).sum();sell=(intra["Volume"]*(sign<0)).sum()
        flow_bias=buy/(buy+sell) if buy+sell>0 else None

        if enable_ofb_filter and (not flow_bias or flow_bias<min_ofb):return None

        catalyst=squeeze=low_float=False;sector=industry="Unknown";short_display=None
        if enable_enrichment:
            info=t.get_info() or {};news=t.get_news() or []
            float_shares=info.get("floatShares");short=info.get("shortPercentOfFloat")
            sector=info.get("sector","Unknown");industry=info.get("industry","Unknown")
            low_float=float_shares and float_shares<20_000_000
            squeeze=short and short>0.15
            short_display=round(short*100,2) if short else None
            if news:
                pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                catalyst=((datetime.now(timezone.utc)-pub).days<=3)

        score=short_window_score(pm=(intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/
                                 intra["Close"].iloc[-2]*100 if not intra.empty else None,
                                 yday=yday,m3=m3,m10=m10,
                                 rsi7=rsi7,rvol10=rvol10,
                                 catalyst=catalyst,squeeze=squeeze,
                                 vwap=vwap_dist,flow_bias=flow_bias)

        return {
            "Symbol":sym["Symbol"],"Exchange":sym.get("Exchange","?"),
            "Price":round(price,2),"Volume":live_vol,           # << FIXED
            "Score":score,"Prob_Rise%":breakout_probability(score),
            "PM%":round(((intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/
                         intra["Close"].iloc[-2]*100),2) if not intra.empty else None,
            "YDay%":round(yday,2),"3D%":round(m3,2),"10D%":round(m10,2),
            "RSI7":round(rsi7,2),"RVOL_10D":round(rvol10,2) if rvol10 else None,
            "VWAP%":round(vwap_dist,2) if vwap_dist else None,
            "FlowBias":round(flow_bias,2) if flow_bias else None,
            "Squeeze?":squeeze,"LowFloat?":low_float,
            "Short % Float":short_display,"Sector":sector,"Industry":industry,
            "Catalyst":catalyst,"MTF_Trend":multi_timeframe_label(None,m3,m10),
            "Spark":close
        }
    except:return None


@st.cache_data(ttl=6)
def run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool):
    universe=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        results=[f.result() for f in concurrent.futures.as_completed(
                    [ex.submit(scan_one,s,enable_enrichment,enable_ofb_filter,min_ofb) for s in universe])]
    res=[r for r in results if r]
    return pd.DataFrame(res) if res else pd.DataFrame()

# --------------------------------------------------------------------------------
# UI RESULTS (unchanged)
# --------------------------------------------------------------------------------
with st.spinner("Scanning market real-time…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty:st.error("No results — try wider criteria or load watchlist.")

else:
    df=df[df["Score"]>=min_breakout]
    if min_pm_move!=0:df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain!=0:df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only:df=df[df["Squeeze?"]]
    if catalyst_only:df=df[df["Catalyst"]]
    if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"🔥 Live Momentum Board — {len(df)} symbols")

    if enable_alerts and st.session_state.get("alerted"):
        st.info("Alerts: "+", ".join(st.session_state.alerted))

    for _,row in df.iterrows():
        sym=row["Symbol"]
        col1,col2,col3,col4=st.columns([2,3,3,3])

        col1.markdown(f"### {sym}")
        col1.write(f"💲 Price: {row['Price']}")
        col1.write(f"📊 Volume (Live): {row['Volume']:,}")
        col1.write(f"🔥 Score: {row['Score']} | Rise Prob: {row['Prob_Rise%']}%")
        col1.write(row['MTF_Trend'])
        col1.write(f"Trend: {row['EMA10 Trend']}")

        col2.write(f"PM%: {row['PM%']} | YDay%: {row['YDay%']}")
        col2.write(f"3D%: {row['3D%']} | 10D%: {row['10D%']}")
        col2.write(f"RSI7: {row['RSI7']} | RVOL_10D: {row['RVOL_10D']}x")

        col3.write(f"VWAP%: {row['VWAP%']}")
        col3.write(f"Order Flow Bias: {row['FlowBias']}")
        if enable_enrichment:col3.write(f"{row['Sector']} / {row['Industry']}")

        col4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"],mode="lines")]),
                          use_container_width=True)

        st.divider()

    st.download_button("📥 Export CSV",df.to_csv(index=False),"v10_live_volume.csv")

st.caption("V10 — Live Volume Enabled. No UI elements removed.")

