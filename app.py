
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Conservative Pro Crypto Scanner V3",
                   page_icon="₿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1500px;padding:.8rem .8rem 2rem}
.signal-box{border:1px solid rgba(148,163,184,.18);border-radius:16px;padding:15px;
background:linear-gradient(145deg,#111827,#0f172a);min-height:210px;margin-bottom:12px}
.buy{border-left:6px solid #22c55e}.sell{border-left:6px solid #ef4444}
.wait{border-left:6px solid #f59e0b}.strongbuy{border-left:6px solid #10b981}
.strongsell{border-left:6px solid #dc2626}
.coin{font-size:1.15rem;font-weight:700}.signal{font-size:1.45rem;font-weight:800;margin:7px 0}
.muted{color:#94a3b8;font-size:.80rem}.price{font-size:1.22rem;font-weight:700}
.tag{display:inline-block;padding:3px 8px;margin:2px;border-radius:99px;background:#1e293b;color:#cbd5e1;font-size:.70rem}
div[data-testid="stMetric"]{background:#0f172a;border:1px solid rgba(148,163,184,.14);padding:10px;border-radius:14px}
@media(max-width:700px){.block-container{padding:.5rem}.signal-box{min-height:190px;padding:12px}}
</style>
""", unsafe_allow_html=True)

COINS={"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD",
       "Solana":"SOL-USD","XRP":"XRP-USD"}

st.title("₿ Conservative Pro Crypto Scanner V3")
st.caption("EMA 200 • RSI 10/14 • Volume • ATR • Trend/Chop Filter • 2% SL • 4% TP")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    interval=st.selectbox("Candle timeframe",["1h","2h","4h","1d"],index=2)
    refresh=st.slider("Auto refresh (seconds)",15,300,30,15)
    auto=st.toggle("🔄 Auto Refresh",True)
    st.divider()
    st.subheader("Core Strategy")
    ema_len=st.number_input("EMA",100,400,200)
    rsi_len=st.number_input("RSI",5,30,10)
    rsi_sma_len=st.number_input("RSI SMA",5,30,14)
    st.subheader("Confirmation")
    volume_len=st.number_input("Volume SMA",5,50,20)
    atr_len=st.number_input("ATR",5,50,14)
    min_atr_pct=st.number_input("Minimum ATR %",0.05,10.0,0.20,0.05)
    st.subheader("Risk")
    sl_pct=st.number_input("Stop Loss %",0.1,20.0,2.0,0.1)
    tp_pct=st.number_input("Take Profit %",0.1,50.0,4.0,0.1)

if auto:
    st_autorefresh(interval=refresh*1000,key="v3_refresh")

@st.cache_data(ttl=20,show_spinner=False)
def fetch(ticker, interval, period):
    native="1h" if interval in ("1h","2h","4h") else "1d"
    raw=yf.download(ticker,period=period,interval=native,progress=False,
                    auto_adjust=False,threads=False)
    if raw is None or raw.empty:return None
    if isinstance(raw.columns,pd.MultiIndex):
        raw.columns=raw.columns.get_level_values(0)
    raw=raw.reset_index()
    tc="Datetime" if "Datetime" in raw.columns else "Date"
    raw=raw.rename(columns={tc:"Date"})
    keep=[c for c in ["Date","Open","High","Low","Close","Volume"] if c in raw.columns]
    raw=raw[keep].copy()
    raw["Date"]=pd.to_datetime(raw["Date"],errors="coerce",utc=True)
    for c in ["Open","High","Low","Close","Volume"]:
        raw[c]=pd.to_numeric(raw[c],errors="coerce")
    raw=raw.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").set_index("Date")
    if interval in ("2h","4h"):
        raw=raw.resample(interval).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return raw

def rsi(close,n):
    d=close.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=loss.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))

def indicators(df):
    d=df.copy()
    d["EMA200"]=d.Close.ewm(span=ema_len,adjust=False).mean()
    d["RSI"]=rsi(d.Close,rsi_len)
    d["RSI_SMA"]=d.RSI.rolling(rsi_sma_len).mean()
    d["VOL_SMA"]=d.Volume.rolling(volume_len).mean()
    prev=d.Close.shift(1)
    tr=pd.concat([(d.High-d.Low),(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(atr_len).mean()
    d["ATR_PCT"]=d.ATR/d.Close*100
    d["EMA_SLOPE"]=d.EMA200-d.EMA200.shift(5)
    d["BullCross"]=(d.RSI>d.RSI_SMA)&(d.RSI.shift(1)<=d.RSI_SMA.shift(1))
    d["BearCross"]=(d.RSI<d.RSI_SMA)&(d.RSI.shift(1)>=d.RSI_SMA.shift(1))
    d["VOL_OK"]=d.Volume>=d.VOL_SMA*0.90
    d["VOL_STRONG"]=d.Volume>=d.VOL_SMA*1.20
    # Stronger entries: trend + momentum + candle + volatility + volume.
    d["LONG_SIGNAL"]=(
        d.BullCross & (d.Close>d.EMA200) & (d.EMA_SLOPE>0) &
        (d.RSI>=52) & (d.RSI<=68) & d.VOL_OK & (d.ATR_PCT>=min_atr_pct) &
        (d.Close>d.Open)
    )
    d["SHORT_SIGNAL"]=(
        d.BearCross & (d.Close<d.EMA200) & (d.EMA_SLOPE<0) &
        (d.RSI<=48) & (d.RSI>=32) & d.VOL_OK & (d.ATR_PCT>=min_atr_pct) &
        (d.Close<d.Open)
    )
    return d.dropna()

def classify(d):
    r=d.iloc[-1]
    long_score=sum([r.Close>r.EMA200,r.EMA_SLOPE>0,r.RSI>r.RSI_SMA,
                    52<=r.RSI<=68,r.Volume>=r.VOL_SMA*.90,r.ATR_PCT>=min_atr_pct,r.Close>r.Open])
    short_score=sum([r.Close<r.EMA200,r.EMA_SLOPE<0,r.RSI<r.RSI_SMA,
                     32<=r.RSI<=48,r.Volume>=r.VOL_SMA*.90,r.ATR_PCT>=min_atr_pct,r.Close<r.Open])
    fresh="BUY" if bool(r.LONG_SIGNAL) else "SELL" if bool(r.SHORT_SIGNAL) else None
    if fresh=="BUY" and long_score>=6:
        signal="STRONG BUY"
    elif fresh=="SELL" and short_score>=6:
        signal="STRONG SELL"
    elif fresh=="BUY": signal="BUY"
    elif fresh=="SELL": signal="SELL"
    elif long_score>=5 and r.Close>r.EMA200 and r.RSI>r.RSI_SMA: signal="BUY"
    elif short_score>=5 and r.Close<r.EMA200 and r.RSI<r.RSI_SMA: signal="SELL"
    else: signal="WAIT"
    entry=float(r.Close)
    if "BUY" in signal: sl=entry*(1-sl_pct/100); tp=entry*(1+tp_pct/100)
    elif "SELL" in signal: sl=entry*(1+sl_pct/100); tp=entry*(1-tp_pct/100)
    else: sl=tp=None
    strength=max(long_score,short_score)/7*100
    cls="strongbuy" if signal=="STRONG BUY" else "buy" if signal=="BUY" else "strongsell" if signal=="STRONG SELL" else "sell" if signal=="SELL" else "wait"
    return {"signal":signal,"cls":cls,"price":entry,"ema":float(r.EMA200),"rsi":float(r.RSI),
            "rsi_sma":float(r.RSI_SMA),"atr":float(r.ATR_PCT),"strength":strength,
            "volume_ok":bool(r.VOL_OK),"time":r.name}

def backtest(df, initial=10000.0):
    d=indicators(df)
    capital=initial; pos=None; entry=0.0; trades=[]
    equity=[]
    for ts,r in d.iterrows():
        hi=float(r.High); lo=float(r.Low); close=float(r.Close)
        if pos is None:
            if bool(r.LONG_SIGNAL): pos="LONG"; entry=close
            elif bool(r.SHORT_SIGNAL): pos="SHORT"; entry=close
        elif pos=="LONG":
            sl=entry*(1-sl_pct/100); tp=entry*(1+tp_pct/100)
            if lo<=sl:
                exitp=sl; reason="SL"
            elif hi>=tp:
                exitp=tp; reason="TP"
            elif bool(r.SHORT_SIGNAL):
                exitp=close; reason="SIGNAL"
            else: exitp=None
            if exitp is not None:
                pnl=capital*(exitp-entry)/entry; capital+=pnl
                trades.append([ts,pos,entry,exitp,pnl,reason]); pos=None
        else:
            sl=entry*(1+sl_pct/100); tp=entry*(1-tp_pct/100)
            if hi>=sl:
                exitp=sl; reason="SL"
            elif lo<=tp:
                exitp=tp; reason="TP"
            elif bool(r.LONG_SIGNAL):
                exitp=close; reason="SIGNAL"
            else: exitp=None
            if exitp is not None:
                pnl=capital*(entry-exitp)/entry; capital+=pnl
                trades.append([ts,pos,entry,exitp,pnl,reason]); pos=None
        mark=capital if pos is None else capital*((close-entry)/entry+1 if pos=="LONG" else (entry-close)/entry+1)
        equity.append((ts,mark))
    if pos and len(d):
        close=float(d.iloc[-1].Close); pnl=capital*((close-entry)/entry if pos=="LONG" else (entry-close)/entry)
        capital+=pnl; trades.append([d.index[-1],pos,entry,close,pnl,"END"])
    t=pd.DataFrame(trades,columns=["Time","Side","Entry","Exit","PnL","Reason"])
    wins=int((t.PnL>0).sum()) if not t.empty else 0
    losses=int((t.PnL<=0).sum()) if not t.empty else 0
    wr=wins/len(t)*100 if len(t) else 0
    eq=pd.DataFrame(equity,columns=["Time","Equity"]).set_index("Time") if equity else pd.DataFrame()
    dd=((eq.Equity-eq.Equity.cummax())/eq.Equity.cummax()*100).min() if not eq.empty else 0
    return capital,capital-initial,len(t),wins,losses,wr,float(dd),t

results={}; errors={}
for name,ticker in COINS.items():
    try:
        raw=fetch(ticker,interval,"90d" if interval!="1d" else "2y")
        if raw is None: continue
        d=indicators(raw)
        if len(d): results[name]={"data":d,"info":classify(d)}
    except Exception as e: errors[name]=str(e)[:100]

st.subheader("📡 Market Overview")
order=["STRONG BUY","BUY","WAIT","SELL","STRONG SELL"]
cnt={k:sum(v["info"]["signal"]==k for v in results.values()) for k in order}
cs=st.columns(5)
for c,k in zip(cs,order): c.metric(k,cnt[k])

st.subheader("🎯 Live Coin Scanner")
names=list(COINS)
for start in range(0,len(names),2):
    cols=st.columns(2)
    for j,name in enumerate(names[start:start+2]):
        with cols[j]:
            if name not in results:
                st.error(f"{name}: unavailable"); continue
            x=results[name]["info"]
            icon="🟢" if "BUY" in x["signal"] else "🔴" if "SELL" in x["signal"] else "🟡"
            en=f"{x['price']:,.6f}" if x["signal"]!="WAIT" else "—"
            sl=f"{x['price']*(1-sl_pct/100):,.6f}" if "BUY" in x["signal"] else f"{x['price']*(1+sl_pct/100):,.6f}" if "SELL" in x["signal"] else "—"
            tp=f"{x['price']*(1+tp_pct/100):,.6f}" if "BUY" in x["signal"] else f"{x['price']*(1-tp_pct/100):,.6f}" if "SELL" in x["signal"] else "—"
            st.markdown(f"""<div class="signal-box {x['cls']}">
<div class="coin">{icon} {name}</div><div class="signal">{x['signal']}</div>
<div class="price">${x['price']:,.6f}</div>
<span class="tag">Strength {x['strength']:.0f}%</span>
<span class="tag">RSI {x['rsi']:.2f}</span>
<span class="tag">ATR {x['atr']:.2f}%</span>
<span class="tag">Volume {'OK' if x['volume_ok'] else 'LOW'}</span>
<div style="margin-top:10px"><b>Entry:</b> {en} &nbsp; <b>SL:</b> {sl} &nbsp; <b>TP:</b> {tp}</div>
</div>""",unsafe_allow_html=True)

st.divider()
st.subheader("🧪 V3 Backtest")
st.caption("30 days • ₹10,000 starting capital • same V3 filters • 2% SL • 4% TP")

if st.button("▶️ Run 30-Day V3 Backtest",use_container_width=True):
    rows=[]; all_trades=[]
    with st.spinner("Backtesting BTC, ETH, BNB, SOL and XRP..."):
        cutoff=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=30)
        for name,ticker in COINS.items():
            try:
                raw=fetch(ticker,interval,"90d" if interval!="1d" else "2y")
                if raw is None: continue
                d=raw[raw.index<=pd.Timestamp.now(tz="UTC")]
                prep=indicators(d)
                test=prep[prep.index>=cutoff]
                if test.empty: continue
                final,pnl,n,w,l,wr,dd,tr=backtest(d[ d.index>=cutoff ],10000.0)
                rows.append({"Coin":name,"Final Balance":final,"P/L":pnl,"Trades":n,
                             "Wins":w,"Losses":l,"Win Rate %":wr,"Max DD %":dd})
                if not tr.empty:
                    tr=tr.copy(); tr.insert(0,"Coin",name); all_trades.append(tr)
            except Exception as e:
                st.warning(f"{name}: {str(e)[:100]}")
    if rows:
        r=pd.DataFrame(rows)
        total_trades=int(r.Trades.sum()); wins=int(r.Wins.sum()); losses=int(r.Losses.sum())
        total_pnl=float(r["P/L"].sum())
        # Each coin is an independent ₹10,000 simulation; this aggregate is clearly labeled.
        st.info("Coin-wise simulations use ₹10,000 independently per coin; aggregate P/L below is the sum of those independent simulations.")
        a,b,c,d,e=st.columns(5)
        a.metric("Total P/L",f"₹{total_pnl:,.2f}")
        b.metric("Aggregate Win Rate",f"{wins/total_trades*100:.2f}%" if total_trades else "0%")
        c.metric("Trades",total_trades); d.metric("Wins",wins); e.metric("Losses",losses)
        st.dataframe(r.round(2),use_container_width=True,hide_index=True)
        if all_trades:
            st.subheader("Trade History")
            st.dataframe(pd.concat(all_trades,ignore_index=True).round(6),use_container_width=True,hide_index=True)
    else:
        st.warning("No valid 30-day backtest data was available.")

st.subheader("🔎 Detailed Chart")
selected=st.selectbox("Select coin",names)
if selected in results:
    d=results[selected]["data"]
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="Price"))
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA200,mode="lines",name="EMA 200"))
    longs=d[d.LONG_SIGNAL]; shorts=d[d.SHORT_SIGNAL]
    if not longs.empty: fig.add_trace(go.Scatter(x=longs.index,y=longs.Low*.995,mode="markers",marker=dict(symbol="triangle-up",size=11),name="BUY"))
    if not shorts.empty: fig.add_trace(go.Scatter(x=shorts.index,y=shorts.High*1.005,mode="markers",marker=dict(symbol="triangle-down",size=11),name="SELL"))
    fig.update_layout(height=600,xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

st.caption("Educational backtest. A 70% win rate or fixed profit is not guaranteed; validate on unseen data before using real money.")
