import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Conservative Pro Crypto Scanner V2",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

COINS = {
    "Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD",
    "Solana":"SOL-USD","XRP":"XRP-USD"
}

st.title("₿ Conservative Pro Crypto Scanner")
st.caption("EMA 200 • RSI(10) • RSI SMA(14) • Long + Short • 2% SL • 4% TP")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    interval_choice = st.selectbox("Candle timeframe", ["1h","2h","4h","1d"], index=2)
    refresh_seconds = st.slider("Auto refresh (seconds)", 15, 300, 30, 15)
    auto_refresh = st.toggle("🔄 Auto Refresh", True)
    st.divider()
    st.subheader("Strategy")
    ema_len = st.number_input("EMA", 10, 500, 200)
    rsi_len = st.number_input("RSI", 2, 100, 10)
    rsi_sma_len = st.number_input("RSI SMA", 2, 100, 14)
    st.divider()
    st.subheader("Risk Management")
    sl_pct = st.number_input("Stop Loss %", 0.1, 20.0, 2.0, 0.1)
    tp_pct = st.number_input("Take Profit %", 0.1, 50.0, 4.0, 0.1)

if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, key="crypto_auto_refresh")

@st.cache_data(ttl=20, show_spinner=False)
def fetch_coin(ticker, interval):
    native = "1h" if interval in ("1h","2h","4h") else "1d"
    period = "60d" if native == "1h" else "2y"
    raw = yf.download(ticker, period=period, interval=native, progress=False,
                      auto_adjust=False, threads=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw = raw.rename(columns={time_col:"Date"})
    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in raw.columns]
    raw = raw[keep].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").set_index("Date")
    if interval in ("2h","4h"):
        raw = raw.resample(interval).agg({
            "Open":"first","High":"max","Low":"min","Close":"last",
            "Volume":"sum" if "Volume" in raw.columns else "last"
        }).dropna(subset=["Open","High","Low","Close"])
    return raw

def rsi_wilder(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    d = df.copy()
    d["EMA"] = d["Close"].ewm(span=ema_len, adjust=False).mean()
    d["RSI"] = rsi_wilder(d["Close"], rsi_len)
    d["RSI_SMA"] = d["RSI"].rolling(rsi_sma_len).mean()
    d["BullCross"] = (d["RSI"] > d["RSI_SMA"]) & (d["RSI"].shift(1) <= d["RSI_SMA"].shift(1))
    d["BearCross"] = (d["RSI"] < d["RSI_SMA"]) & (d["RSI"].shift(1) >= d["RSI_SMA"].shift(1))
    d["LONG_SIGNAL"] = d["BullCross"] & (d["Close"] > d["EMA"])
    d["SHORT_SIGNAL"] = d["BearCross"] & (d["Close"] < d["EMA"])
    return d.dropna()

def last_signal(d):
    longs = d.index[d["LONG_SIGNAL"]]
    shorts = d.index[d["SHORT_SIGNAL"]]
    if len(longs) == 0 and len(shorts) == 0:
        return None, None
    lt = longs[-1] if len(longs) else pd.Timestamp.min.tz_localize("UTC")
    stt = shorts[-1] if len(shorts) else pd.Timestamp.min.tz_localize("UTC")
    return ("BUY", lt) if lt > stt else ("SELL", stt)

def build_info(d):
    r = d.iloc[-1]
    price = float(r["Close"]); ema = float(r["EMA"])
    rsi = float(r["RSI"]); rsisma = float(r["RSI_SMA"])
    trend_up = price > ema
    momentum_up = rsi > rsisma
    gap = abs(price-ema)/ema*100

    # Fresh signal on current candle
    fresh = "BUY" if bool(r["LONG_SIGNAL"]) else "SELL" if bool(r["SHORT_SIGNAL"]) else None

    # Persistent active direction: current trend + momentum.
    # This prevents the dashboard from staying WAIT after a previous confirmed signal.
    if trend_up and momentum_up:
        base = "BUY"
        reason = "Price above EMA 200 and RSI above RSI SMA."
    elif (not trend_up) and (not momentum_up):
        base = "SELL"
        reason = "Price below EMA 200 and RSI below RSI SMA."
    else:
        base = "WAIT"
        reason = "Trend and momentum are not aligned."

    if fresh == "BUY" and rsi >= 55 and gap >= 0.25:
        signal = "STRONG BUY"
    elif fresh == "SELL" and rsi <= 45 and gap >= 0.25:
        signal = "STRONG SELL"
    elif fresh == "BUY":
        signal = "BUY"
    elif fresh == "SELL":
        signal = "SELL"
    elif base == "BUY":
        signal = "BUY"
    elif base == "SELL":
        signal = "SELL"
    else:
        signal = "WAIT"

    # Last confirmed crossover and its entry. Used for visibility/persistence.
    ls, ls_time = last_signal(d)
    if signal in ("BUY","STRONG BUY"):
        entry = price
        sl = entry*(1-sl_pct/100)
        tp = entry*(1+tp_pct/100)
    elif signal in ("SELL","STRONG SELL"):
        entry = price
        sl = entry*(1+sl_pct/100)
        tp = entry*(1-tp_pct/100)
    else:
        entry = sl = tp = None

    score = 0
    score += 1 if trend_up else -1
    score += 1 if momentum_up else -1
    score += 1 if rsi >= 55 else -1 if rsi <= 45 else 0
    score += 1 if gap >= 0.25 and trend_up else -1 if gap >= 0.25 and not trend_up else 0
    strength = int(round((score + 4) / 8 * 100))

    cls = "strongbuy" if signal=="STRONG BUY" else "buy" if signal=="BUY" else "strongsell" if signal=="STRONG SELL" else "sell" if signal=="SELL" else "wait"

    return {
        "signal":signal,"class":cls,"price":price,"ema":ema,"rsi":rsi,"rsi_sma":rsisma,
        "trend":"BULLISH" if trend_up else "BEARISH",
        "momentum":"BULLISH" if momentum_up else "BEARISH",
        "entry":entry,"sl":sl,"tp":tp,"score":strength,
        "last_signal":ls,"last_signal_time":ls_time,"reason":reason,"candle_time":r.name
    }

results, errors = {}, {}
for name,ticker in COINS.items():
    try:
        raw = fetch_coin(ticker, interval_choice)
        needed = max(ema_len, rsi_len+rsi_sma_len)+5
        if raw is None or len(raw) < needed:
            errors[name] = "Not enough market data"; continue
        d = add_indicators(raw)
        if len(d) < 2:
            errors[name] = "Not enough indicator data"; continue
        results[name] = {"ticker":ticker,"data":d,"info":build_info(d)}
    except Exception as ex:
        errors[name] = str(ex)[:100]

# ---------------- MARKET OVERVIEW ----------------
st.subheader("📡 Market Overview")
order = ["STRONG BUY","BUY","WAIT","SELL","STRONG SELL"]
counts = {k: sum(v["info"]["signal"] == k for v in results.values()) for k in order}
mc = st.columns(5)
for c,k in zip(mc,order):
    c.metric(k, counts[k])

bull = sum(v["info"]["trend"]=="BULLISH" for v in results.values())
bear = sum(v["info"]["trend"]=="BEARISH" for v in results.values())
neutral = len(results)-bull-bear
market_bias = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"

h1,h2,h3,h4 = st.columns(4)
h1.metric("Market Bias", market_bias)
h2.metric("Bullish Coins", f"{bull}/{len(results)}")
h3.metric("Bearish Coins", f"{bear}/{len(results)}")
h4.metric("Neutral", f"{neutral}/{len(results)}")

if errors:
    st.warning(" • ".join(f"{k}: {v}" for k,v in errors.items()))

# ---------------- PRIORITY ----------------
st.subheader("🔥 Signal Priority")
rank = {"STRONG BUY":6,"BUY":5,"STRONG SELL":-6,"SELL":-5,"WAIT":0}
rows=[]
for name,v in results.items():
    x=v["info"]
    rows.append({
        "Coin":name,"Signal":x["signal"],"Strength":f"{x['score']}%",
        "Price":x["price"],"Trend":x["trend"],"RSI":round(x["rsi"],2),
        "EMA 200":x["ema"],
        "Last Signal":x["last_signal"] or "—",
        "Last Signal Time":str(x["last_signal_time"])[:19] if x["last_signal_time"] else "—"
    })
rows.sort(key=lambda r: rank.get(r["Signal"],0), reverse=True)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------- COIN CARDS ----------------
st.subheader("🎯 Live Coin Scanner")
names=list(COINS.keys())
for start in range(0,len(names),2):
    cols=st.columns(2)
    for j,name in enumerate(names[start:start+2]):
        with cols[j]:
            if name not in results:
                st.error(f"{name}: unavailable"); continue
            x=results[name]["info"]
            icon="🟢" if "BUY" in x["signal"] else "🔴" if "SELL" in x["signal"] else "🟡"
            en="—" if x["entry"] is None else f"{x['entry']:,.6f}"
            sl="—" if x["sl"] is None else f"{x['sl']:,.6f}"
            tp="—" if x["tp"] is None else f"{x['tp']:,.6f}"
            lst="—" if x["last_signal_time"] is None else str(x["last_signal_time"])[:19]
            st.markdown(f"""
<div class="signal-box {x['class']}">
<div class="coin">{icon} {name} <span class="muted">({COINS[name]})</span></div>
<div class="signal">{x['signal']}</div>
<div class="price">${x['price']:,.6f}</div>
<div class="muted">{x['reason']}</div>
<span class="tag">Strength {x['score']}%</span>
<span class="tag">Trend {x['trend']}</span>
<span class="tag">RSI {x['rsi']:.2f}</span>
<span class="tag">RSI SMA {x['rsi_sma']:.2f}</span>
<span class="tag">EMA 200 {x['ema']:,.6f}</span>
<div style="margin-top:10px"><b>Entry:</b> {en} &nbsp; <b>SL:</b> {sl} &nbsp; <b>TP:</b> {tp}</div>
<div class="muted" style="margin-top:8px">Last signal: {lst} UTC</div>
</div>
""", unsafe_allow_html=True)

# ---------------- DETAIL ----------------
st.subheader("🔎 Detailed Chart")
selected=st.selectbox("Select coin",names)
if selected in results:
    d=results[selected]["data"]; x=results[selected]["info"]
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Signal",x["signal"]); c2.metric("Strength",f"{x['score']}%")
    c3.metric("Price",f"${x['price']:,.6f}"); c4.metric("RSI",f"{x['rsi']:.2f}"); c5.metric("Trend",x["trend"])

    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="Price"))
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA,mode="lines",name=f"EMA {ema_len}"))
    longs=d[d.LONG_SIGNAL]; shorts=d[d.SHORT_SIGNAL]
    if not longs.empty:
        fig.add_trace(go.Scatter(x=longs.index,y=longs.Low*.995,mode="markers",
                                 marker=dict(symbol="triangle-up",size=11),name="BUY"))
    if not shorts.empty:
        fig.add_trace(go.Scatter(x=shorts.index,y=shorts.High*1.005,mode="markers",
                                 marker=dict(symbol="triangle-down",size=11),name="SELL"))
    if x["entry"] is not None:
        fig.add_hline(y=x["entry"],line_dash="dot",annotation_text="Entry")
        fig.add_hline(y=x["sl"],line_dash="dash",annotation_text=f"{sl_pct}% SL")
        fig.add_hline(y=x["tp"],line_dash="dash",annotation_text=f"{tp_pct}% TP")
    fig.update_layout(height=600,xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    rf=go.Figure()
    rf.add_trace(go.Scatter(x=d.index,y=d.RSI,mode="lines",name=f"RSI {rsi_len}"))
    rf.add_trace(go.Scatter(x=d.index,y=d.RSI_SMA,mode="lines",name=f"RSI SMA {rsi_sma_len}"))
    rf.add_hline(y=70,line_dash="dash"); rf.add_hline(y=30,line_dash="dash")
    rf.update_layout(height=330,hovermode="x unified")
    st.plotly_chart(rf,use_container_width=True)

st.divider()
st.caption("Educational scanner. Strong BUY/SELL requires a fresh crossover plus RSI/EMA confirmation. Active BUY/SELL remains visible when trend and momentum stay aligned. SL/TP are rule-based targets, not guaranteed execution prices.")
