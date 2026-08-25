
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Conservative Pro Crypto Scanner", page_icon="₿",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1500px;padding:.8rem .8rem 2rem}
.signal-box{border:1px solid rgba(148,163,184,.18);border-radius:16px;padding:15px;
background:linear-gradient(145deg,#111827,#0f172a);min-height:205px;margin-bottom:12px}
.buy{border-left:6px solid #22c55e}.sell{border-left:6px solid #ef4444}
.wait{border-left:6px solid #f59e0b}.strongbuy{border-left:6px solid #10b981}
.strongsell{border-left:6px solid #dc2626}
.coin{font-size:1.15rem;font-weight:700}.signal{font-size:1.45rem;font-weight:800;margin:7px 0}
.muted{color:#94a3b8;font-size:.80rem}.price{font-size:1.22rem;font-weight:700}
.tag{display:inline-block;padding:3px 8px;margin:2px;border-radius:99px;background:#1e293b;color:#cbd5e1;font-size:.70rem}
div[data-testid="stMetric"]{background:#0f172a;border:1px solid rgba(148,163,184,.14);padding:10px;border-radius:14px}
@media(max-width:700px){.block-container{padding:.5rem}.signal-box{min-height:185px;padding:12px}}
</style>
""", unsafe_allow_html=True)

COINS={"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD",
       "Solana":"SOL-USD","XRP":"XRP-USD"}

st.title("₿ Conservative Pro Crypto Scanner")
st.caption("EMA 200 • RSI(10) • RSI SMA(14) • Long + Short • 2% SL • 4% TP")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    interval_choice=st.selectbox("Candle timeframe",["1h","2h","4h","1d"],index=2)
    refresh_seconds=st.slider("Auto refresh (seconds)",15,300,30,15)
    auto_refresh=st.toggle("🔄 Auto Refresh",True)
    st.divider()
    st.subheader("Strategy")
    ema_len=st.number_input("EMA",10,500,200)
    rsi_len=st.number_input("RSI",2,100,10)
    rsi_sma_len=st.number_input("RSI SMA",2,100,14)
    st.divider()
    st.subheader("Risk Management")
    sl_pct=st.number_input("Stop Loss %",0.1,20.0,2.0,0.1)
    tp_pct=st.number_input("Take Profit %",0.1,50.0,4.0,0.1)

if auto_refresh:
    st_autorefresh(interval=refresh_seconds*1000,key="crypto_auto_refresh")

@st.cache_data(ttl=20,show_spinner=False)
def download_data(ticker,interval,period):
    native="1h" if interval in ("1h","2h","4h") else "1d"
    raw=yf.download(ticker,period=period,interval=native,progress=False,
                    auto_adjust=False,threads=False)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex):
        raw.columns=raw.columns.get_level_values(0)
    raw=raw.reset_index()
    tc="Datetime" if "Datetime" in raw.columns else "Date"
    raw=raw.rename(columns={tc:"Date"})
    keep=[c for c in ["Date","Open","High","Low","Close","Volume"] if c in raw.columns]
    raw=raw[keep].copy()
    raw["Date"]=pd.to_datetime(raw["Date"],errors="coerce",utc=True)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in raw.columns: raw[c]=pd.to_numeric(raw[c],errors="coerce")
    raw=raw.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").set_index("Date")
    if interval in ("2h","4h"):
        raw=raw.resample(interval).agg({
            "Open":"first","High":"max","Low":"min","Close":"last",
            "Volume":"sum" if "Volume" in raw.columns else "last"
        }).dropna(subset=["Open","High","Low","Close"])
    return raw

def rsi_wilder(close,length):
    delta=close.diff()
    gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/length,adjust=False,min_periods=length).mean()
    al=loss.ewm(alpha=1/length,adjust=False,min_periods=length).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))

def indicators(df):
    d=df.copy()
    d["EMA"]=d["Close"].ewm(span=ema_len,adjust=False).mean()
    d["RSI"]=rsi_wilder(d["Close"],rsi_len)
    d["RSI_SMA"]=d["RSI"].rolling(rsi_sma_len).mean()
    d["BullCross"]=(d.RSI>d.RSI_SMA)&(d.RSI.shift(1)<=d.RSI_SMA.shift(1))
    d["BearCross"]=(d.RSI<d.RSI_SMA)&(d.RSI.shift(1)>=d.RSI_SMA.shift(1))
    d["LONG_SIGNAL"]=d.BullCross&(d.Close>d.EMA)
    d["SHORT_SIGNAL"]=d.BearCross&(d.Close<d.EMA)
    return d.dropna()

def signal_info(d):
    r=d.iloc[-1]
    fresh="BUY" if bool(r.LONG_SIGNAL) else "SELL" if bool(r.SHORT_SIGNAL) else None
    trend_up=bool(r.Close>r.EMA); momentum_up=bool(r.RSI>r.RSI_SMA)
    gap=abs(float(r.Close-r.EMA))/float(r.EMA)*100
    if fresh=="BUY" and r.RSI>=55 and gap>=.25: signal="STRONG BUY"
    elif fresh=="SELL" and r.RSI<=45 and gap>=.25: signal="STRONG SELL"
    elif fresh: signal=fresh
    elif trend_up and momentum_up: signal="BUY"
    elif not trend_up and not momentum_up: signal="SELL"
    else: signal="WAIT"
    entry=float(r.Close)
    if "BUY" in signal: sl=entry*(1-sl_pct/100); tp=entry*(1+tp_pct/100)
    elif "SELL" in signal: sl=entry*(1+sl_pct/100); tp=entry*(1-tp_pct/100)
    else: sl=tp=None
    score=int(round(((
        (1 if trend_up else -1)+(1 if momentum_up else -1)+
        (1 if r.RSI>=55 else -1 if r.RSI<=45 else 0)+
        (1 if gap>=.25 and trend_up else -1 if gap>=.25 else 0)
    )+4)/8*100))
    cls="strongbuy" if signal=="STRONG BUY" else "buy" if signal=="BUY" else "strongsell" if signal=="STRONG SELL" else "sell" if signal=="SELL" else "wait"
    return dict(signal=signal,cls=cls,price=entry,ema=float(r.EMA),rsi=float(r.RSI),
                rsi_sma=float(r.RSI_SMA),trend="BULLISH" if trend_up else "BEARISH",
                momentum="BULLISH" if momentum_up else "BEARISH",entry=entry if signal!="WAIT" else None,
                sl=sl,tp=tp,score=score,time=r.name)

# Backtest requested by the user: fixed starting capital ₹10,000, last 30 days.
def run_backtest(df, initial_capital=10000.0):
    d=indicators(df).copy()
    capital=float(initial_capital)
    position=None
    entry_price=0.0
    trades=[]
    equity=[]

    for ts,row in d.iterrows():
        high=float(row.High); low=float(row.Low); close=float(row.Close)

        # Entry only on a confirmed strategy signal.
        if position is None:
            if bool(row.LONG_SIGNAL):
                position="LONG"; entry_price=close
            elif bool(row.SHORT_SIGNAL):
                position="SHORT"; entry_price=close
        else:
            if position=="LONG":
                sl=entry_price*(1-sl_pct/100); tp=entry_price*(1+tp_pct/100)
                hit=None; exit_price=None
                # Conservative assumption if both are touched in one candle: SL first.
                if low<=sl: hit="SL"; exit_price=sl
                elif high>=tp: hit="TP"; exit_price=tp
                elif bool(row.SHORT_SIGNAL):
                    hit="SIGNAL"; exit_price=close
                if hit:
                    pnl_pct=(exit_price-entry_price)/entry_price*100
                    pnl=capital*(pnl_pct/100)
                    capital+=pnl
                    trades.append([ts,"LONG",entry_price,exit_price,pnl,hit])
                    position=None
            else:
                sl=entry_price*(1+sl_pct/100); tp=entry_price*(1-tp_pct/100)
                hit=None; exit_price=None
                if high>=sl: hit="SL"; exit_price=sl
                elif low<=tp: hit="TP"; exit_price=tp
                elif bool(row.LONG_SIGNAL):
                    hit="SIGNAL"; exit_price=close
                if hit:
                    pnl_pct=(entry_price-exit_price)/entry_price*100
                    pnl=capital*(pnl_pct/100)
                    capital+=pnl
                    trades.append([ts,"SHORT",entry_price,exit_price,pnl,hit])
                    position=None

        # Mark-to-market equity for drawdown calculation.
        if position=="LONG":
            eq=capital*(1+(close-entry_price)/entry_price)
        elif position=="SHORT":
            eq=capital*(1+(entry_price-close)/entry_price)
        else:
            eq=capital
        equity.append((ts,eq))

    # Close any open position at the last available close for reporting.
    if position is not None and len(d):
        last_ts=d.index[-1]; last_close=float(d.iloc[-1].Close)
        if position=="LONG":
            pnl=capital*((last_close-entry_price)/entry_price)
        else:
            pnl=capital*((entry_price-last_close)/entry_price)
        capital+=pnl
        trades.append([last_ts,position,entry_price,last_close,pnl,"END"])

    tdf=pd.DataFrame(trades,columns=["Time","Side","Entry","Exit","PnL","Exit Reason"])
    if tdf.empty:
        wins=losses=0; win_rate=0.0
    else:
        wins=int((tdf.PnL>0).sum()); losses=int((tdf.PnL<=0).sum())
        win_rate=wins/len(tdf)*100

    eqdf=pd.DataFrame(equity,columns=["Time","Equity"]).set_index("Time") if equity else pd.DataFrame()
    if not eqdf.empty:
        peak=eqdf.Equity.cummax()
        dd=(eqdf.Equity-peak)/peak*100
        max_dd=float(dd.min())
    else:
        max_dd=0.0

    return {
        "final":capital,"profit":capital-initial_capital,"trades":len(tdf),
        "wins":wins,"losses":losses,"win_rate":win_rate,
        "max_dd":max_dd,"trades_df":tdf,"equity_df":eqdf
    }

# Live scanner data
results={}; errors={}
for name,ticker in COINS.items():
    try:
        raw=download_data(ticker,interval_choice,"60d" if interval_choice!="1d" else "2y")
        if raw is None or len(raw)<max(ema_len,rsi_len+rsi_sma_len)+5:
            errors[name]="Not enough market data"; continue
        d=indicators(raw)
        if len(d)<2: errors[name]="Not enough indicator data"; continue
        results[name]={"data":d,"info":signal_info(d)}
    except Exception as e:
        errors[name]=str(e)[:100]

st.subheader("📡 Market Overview")
orders=["STRONG BUY","BUY","WAIT","SELL","STRONG SELL"]
counts={k:sum(v["info"]["signal"]==k for v in results.values()) for k in orders}
cols=st.columns(5)
for c,k in zip(cols,orders): c.metric(k,counts[k])
bull=sum(v["info"]["trend"]=="BULLISH" for v in results.values())
bear=sum(v["info"]["trend"]=="BEARISH" for v in results.values())
bias="BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
a,b,c=st.columns(3); a.metric("Market Bias",bias); b.metric("Bullish Coins",f"{bull}/{len(results)}"); c.metric("Bearish Coins",f"{bear}/{len(results)}")

if errors: st.warning(" • ".join(f"{k}: {v}" for k,v in errors.items()))

st.subheader("🔥 Signal Priority")
rank={"STRONG BUY":6,"BUY":5,"SELL":-5,"STRONG SELL":-6,"WAIT":0}
rows=[]
for name,v in results.items():
    x=v["info"]
    rows.append({"Coin":name,"Signal":x["signal"],"Strength":f"{x['score']}%",
                 "Price":x["price"],"Trend":x["trend"],"RSI":round(x["rsi"],2)})
rows.sort(key=lambda r:rank.get(r["Signal"],0),reverse=True)
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

st.subheader("🎯 Live Coin Scanner")
names=list(COINS)
for start in range(0,len(names),2):
    cs=st.columns(2)
    for j,name in enumerate(names[start:start+2]):
        with cs[j]:
            if name not in results: st.error(f"{name}: unavailable"); continue
            x=results[name]["info"]
            icon="🟢" if "BUY" in x["signal"] else "🔴" if "SELL" in x["signal"] else "🟡"
            en="—" if x["entry"] is None else f"{x['entry']:,.6f}"
            sl="—" if x["sl"] is None else f"{x['sl']:,.6f}"
            tp="—" if x["tp"] is None else f"{x['tp']:,.6f}"
            st.markdown(f"""
<div class="signal-box {x['cls']}">
<div class="coin">{icon} {name} <span class="muted">({COINS[name]})</span></div>
<div class="signal">{x['signal']}</div>
<div class="price">${x['price']:,.6f}</div>
<span class="tag">Strength {x['score']}%</span><span class="tag">Trend {x['trend']}</span>
<span class="tag">RSI {x['rsi']:.2f}</span><span class="tag">EMA {x['ema']:,.6f}</span>
<div style="margin-top:10px"><b>Entry:</b> {en} &nbsp; <b>SL:</b> {sl} &nbsp; <b>TP:</b> {tp}</div>
</div>""",unsafe_allow_html=True)

# ONLY requested new feature: 30-day backtest with ₹10,000 starting capital.
st.divider()
st.subheader("🧪 30-Day Backtest")
st.caption("Starting capital: ₹10,000 • Uses the same BUY/SELL crossover rules • 2% SL • 4% TP")

if st.button("▶️ Run Backtest",use_container_width=True):
    all_trades=[]
    total_initial=10000.0
    per_coin=[]
    with st.spinner("Running 30-day backtest for BTC, ETH, BNB, SOL and XRP..."):
        for name,ticker in COINS.items():
            try:
                raw=download_data(ticker,interval_choice,"60d" if interval_choice!="1d" else "2y")
                if raw is None: continue
                cutoff=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=30)
                raw30=raw[raw.index>=cutoff].copy()
                if len(raw30)<max(ema_len,rsi_len+rsi_sma_len)+5: continue
                res=run_backtest(raw30,total_initial)
                per_coin.append({
                    "Coin":name,"Final Balance":res["final"],"P/L":res["profit"],
                    "Trades":res["trades"],"Wins":res["wins"],"Losses":res["losses"],
                    "Win Rate %":res["win_rate"],"Max DD %":res["max_dd"]
                })
                if not res["trades_df"].empty:
                    t=res["trades_df"].copy(); t.insert(0,"Coin",name); all_trades.append(t)
            except Exception as e:
                st.warning(f"{name}: backtest error — {str(e)[:100]}")

    if per_coin:
        pdf=pd.DataFrame(per_coin)
        # Combined result treats each coin as an independent ₹10,000 test,
        # then reports the sum of P/L and aggregate win rate.
        total_trades=int(pdf.Trades.sum()); total_wins=int(pdf.Wins.sum())
        total_losses=int(pdf.Losses.sum())
        combined_final=total_initial+float(pdf["P/L"].sum())
        combined_profit=combined_final-total_initial
        combined_wr=(total_wins/total_trades*100) if total_trades else 0.0

        q1,q2,q3,q4,q5=st.columns(5)
        q1.metric("Final Balance",f"₹{combined_final:,.2f}")
        q2.metric("Total P/L",f"₹{combined_profit:,.2f}")
        q3.metric("Win Rate",f"{combined_wr:.2f}%")
        q4.metric("Total Trades",total_trades)
        q5.metric("Wins / Losses",f"{total_wins} / {total_losses}")

        st.dataframe(pdf.round({"Final Balance":2,"P/L":2,"Win Rate %":2,"Max DD %":2}),
                     use_container_width=True,hide_index=True)

        if all_trades:
            st.subheader("Trade History")
            st.dataframe(pd.concat(all_trades,ignore_index=True).round(6),
                         use_container_width=True,hide_index=True)
    else:
        st.info("No sufficient 30-day data/trades were available.")

st.subheader("🔎 Detailed Chart")
selected=st.selectbox("Select coin",names)
if selected in results:
    d=results[selected]["data"]; x=results[selected]["info"]
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="Price"))
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA,mode="lines",name=f"EMA {ema_len}"))
    longs=d[d.LONG_SIGNAL]; shorts=d[d.SHORT_SIGNAL]
    if not longs.empty: fig.add_trace(go.Scatter(x=longs.index,y=longs.Low*.995,mode="markers",marker=dict(symbol="triangle-up",size=11),name="BUY"))
    if not shorts.empty: fig.add_trace(go.Scatter(x=shorts.index,y=shorts.High*1.005,mode="markers",marker=dict(symbol="triangle-down",size=11),name="SELL"))
    fig.update_layout(height=600,xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

st.caption("Educational scanner. Backtest results are historical simulation results and are not guaranteed future returns.")
